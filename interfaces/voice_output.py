"""Voice synthesis pipeline using optional Piper backend."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


try:
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

import yaml


def get_setting(*keys: str, default=None):
    """Local copy of get_setting to avoid circular imports."""
    from pathlib import Path

    settings_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    try:
        with open(settings_path, "r") as f:
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


class VoiceSynthesisUnavailable(RuntimeError):
    """Raised when the system cannot synthesize voice output."""


@dataclass
class VoiceOutput:
    sample_rate: int = 22_050
    cache_dir: Path = Path("storage/voice_cache")

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.backend = get_setting("voice", "tts_backend", default="piper")
        self.binary = Path(get_setting("voice", "tts_binary", default="piper"))
        self.voice_model = get_setting("voice", "tts_voice", default="")

    def speak(self, text: str) -> None:
        audio_path = self._synthesize(text)
        if audio_path is None:
            print(f"Ada (text): {text}")
            return
        if sd is None or sf is None:
            print(f"[voice disabled] Ada: {text}")
            return
        data, samplerate = sf.read(audio_path, dtype="float32")
        sd.play(data, samplerate)
        sd.wait()

    def _synthesize(self, text: str) -> Optional[Path]:
        if self.backend != "piper" or not self.binary.exists():
            return None
        if not self.voice_model:
            return None
        output_path = self._cache_path(text)
        if output_path.exists():
            return output_path
        command = [
            str(self.binary),
            "--model",
            str(self.voice_model),
            "--output_file",
            str(output_path),
        ]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert process.stdin is not None
            process.stdin.write(text.encode("utf-8"))
            process.stdin.close()
            process.wait()
        except (FileNotFoundError, OSError):
            return None
        if not output_path.exists():
            return None
        return output_path

    def _cache_path(self, text: str) -> Path:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"ada_{digest}.wav"
