"""Speech input pipeline leveraging optional Whisper.cpp transcription."""

from __future__ import annotations

import queue
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterable, Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

import torch

import yaml

def get_setting(*keys: str, default=None):
    """Local copy of get_setting to avoid circular imports."""
    from pathlib import Path
    settings_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    try:
        with open(settings_path, 'r') as f:
            settings = yaml.safe_load(f)
        
        node = settings
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node
    except (FileNotFoundError, yaml.YAMLError):
        return default


class VoiceCaptureUnavailable(RuntimeError):
    """Raised when the local system is missing audio dependencies."""


class VoiceActivityDetector:
    """Leverages Silero VAD when available, otherwise falls back to energy threshold."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._threshold = 0.01
        self._silero_model: Optional[torch.nn.Module] = None
        self._load_silero()

    def _load_silero(self) -> None:
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
        try:
            bundle: Any = torch.hub.load("snakers4/silero-vad", model="silero_vad", source="github")
        except Exception:
            self._silero_model = None
            return
        model: Optional[torch.nn.Module]
        if isinstance(bundle, tuple):
            model = bundle[0]
        else:
            model = bundle
        if isinstance(model, torch.nn.Module):
            model.to(device)
            model.eval()
            self._silero_model = model
        else:
            self._silero_model = None

    def is_speech(self, audio: np.ndarray) -> bool:
        if self._silero_model is not None:
            audio_tensor = torch.from_numpy(audio.copy()).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            with torch.no_grad():
                probability = self._silero_model(audio_tensor, self.sample_rate)
            return bool(probability.squeeze().item() > 0.5)
        energy = float(np.mean(np.square(audio)))
        return energy > self._threshold


@dataclass
class TranscriptionResult:
    text: str
    audio_path: Path
    audio_data: np.ndarray


class WhisperCppTranscriber:
    """Wrapper around the whisper.cpp binary."""

    def __init__(self) -> None:
        self.binary = Path(get_setting("voice", "stt_binary", default="whisper"))
        self.model = get_setting("voice", "stt_model", default="")

    def available(self) -> bool:
        return self.binary.exists()

    def transcribe(self, wav_path: Path) -> str:
        if not self.available():
            return ""
        output_dir = wav_path.parent
        timestamp = int(time.time() * 1000)
        output_path = output_dir / f"transcript_{timestamp}"
        command = [
            str(self.binary),
            "-m",
            str(self.model),
            "-f",
            str(wav_path),
            "-otxt",
            "-of",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""
        txt_path = output_path.with_suffix(".txt")
        if not txt_path.exists():
            return ""
        return txt_path.read_text(encoding="utf-8").strip()


class SpeechInput:
    """Captures microphone audio, segments utterances, and transcribes them."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        block_duration: float = 0.5,
        silence_blocks: int = 4,
    ) -> None:
        if sd is None or sf is None:
            raise VoiceCaptureUnavailable("sounddevice and soundfile packages are required for speech capture")
        self.sample_rate = sample_rate
        self.block_size = int(sample_rate * block_duration)
        self.silence_blocks = silence_blocks
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self.vad = VoiceActivityDetector(sample_rate)
        self.transcriber = WhisperCppTranscriber()

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:  # pragma: no cover
        if status:
            print(status)
        self._queue.put(indata.copy())

    def phrases(self) -> Generator[TranscriptionResult, None, None]:
        if sd is None:
            raise VoiceCaptureUnavailable("sounddevice not available")
        stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            dtype="float32",
            blocksize=self.block_size,
            callback=self._callback,
        )
        buffer: list[np.ndarray] = []
        silence_counter = 0
        speaking = False
        with stream:
            while True:
                chunk = self._queue.get()
                chunk = chunk.flatten()
                is_speech = self.vad.is_speech(chunk)
                if is_speech:
                    buffer.append(chunk)
                    silence_counter = 0
                    speaking = True
                elif speaking:
                    silence_counter += 1
                    if silence_counter >= self.silence_blocks:
                        yield self._finalize(buffer)
                        buffer = []
                        silence_counter = 0
                        speaking = False

    def _finalize(self, frames: Iterable[np.ndarray]) -> TranscriptionResult:
        audio = np.concatenate(list(frames))
        tmp_dir = Path(tempfile.mkdtemp(prefix="ada_voice_"))
        wav_path = tmp_dir / "utterance.wav"
        assert sf is not None
        sf.write(wav_path, audio, self.sample_rate)
        text = self.transcriber.transcribe(wav_path)
        return TranscriptionResult(text=text, audio_path=wav_path, audio_data=audio)
