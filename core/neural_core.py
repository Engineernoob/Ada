"""High-level conversational interface for Ada."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from .dialogue_manager import process_conversation

try:
    from .reasoning import ReasoningEngine
except Exception as exc:  # pragma: no cover - surface import issues early
    raise ImportError("ReasoningEngine is required for AdaCore") from exc

try:  # Optional voice synthesis integration
    from voice.synthesize import generate_speech
except ImportError:  # pragma: no cover - voice module optional
    generate_speech = None  # type: ignore


PERSONALITY_PATH = Path(__file__).resolve().parent / "personality.yaml"


def _load_personality() -> Dict[str, Any]:
    if not PERSONALITY_PATH.exists():
        return {}
    try:
        with PERSONALITY_PATH.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
            return data if isinstance(data, dict) else {}
    except (yaml.YAMLError, OSError):
        return {}


@dataclass
class AdaCore:
    model: ReasoningEngine
    personality: Dict[str, Any]

    @classmethod
    def load(cls, checkpoint_path: str | Path | None = None) -> "AdaCore":
        path = Path(checkpoint_path) if checkpoint_path else None
        engine = ReasoningEngine(checkpoint_path=path, use_persona=False)
        personality = _load_personality()
        return cls(model=engine, personality=personality)

    def infer(self, prompt: str) -> Tuple[str, float, str]:
        response, confidence, tone = process_conversation(prompt, self.model)
        self._maybe_speak(response)
        print(f"🗣️ Ada ({tone}): {response}")
        return response, confidence, tone

    def _maybe_speak(self, response: str) -> None:
        if generate_speech is None:
            return
        voice_id = self._voice_id()
        if not voice_id:
            return

        output_path = Path("logs") / "temp_response.wav"
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            generate_speech(voice_id, response, str(output_path))
        except Exception:
            return

    def _voice_id(self) -> Optional[str]:
        personality_cfg = self.personality.get("personality")
        if isinstance(personality_cfg, dict):
            voice = personality_cfg.get("voice")
            if isinstance(voice, str) and voice.strip():
                return voice.strip()
        return None
