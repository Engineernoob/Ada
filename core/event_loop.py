"""Event loops for voice-driven interactions with Ada."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.table import Table

from rl import VoiceReward


console = Console()


@dataclass
class VoiceLoopState:
    status: str = "idle"
    transcript: str = ""
    response: str = ""
    tone: str = "neutral"
    reward: float = 0.0


class _TranscriptionProtocol(Protocol):
    text: str
    audio_path: Path
    audio_data: np.ndarray


class VoiceInteractionLoop:
    def __init__(self) -> None:
        self.state = VoiceLoopState()

    def _render(self) -> Table:
        table = Table(title="Ada Voice Interaction", expand=True)
        table.add_column("Status", justify="left")
        table.add_column("Details", justify="left")
        table.add_row("🎤", self.state.transcript or "Listening...")
        table.add_row("🧠", self.state.response or "Waiting for response")
        table.add_row("🎚", f"Reward {self.state.reward:+.2f} (Tone: {self.state.tone})")
        table.add_row("Mode", self.state.status)
        return table

    def run(
        self,
        capture_iter: Iterable[_TranscriptionProtocol],
        handler: Callable[[str, Path, np.ndarray], tuple[str, VoiceReward]],
        speaker: Callable[[str], None],
    ) -> None:
        with Live(console=console, refresh_per_second=5) as live:
            live.update(self._render())
            try:
                for result in capture_iter:
                    transcript = result.text.strip()
                    if not transcript:
                        continue
                    self.state.status = "processing"
                    self.state.transcript = transcript
                    live.update(self._render())
                    response, voice_reward = handler(
                        transcript, result.audio_path, result.audio_data
                    )
                    self.state.response = response
                    self.state.reward = voice_reward.reward
                    self.state.tone = voice_reward.tone
                    live.update(self._render())
                    speaker(response)
                    self.state.status = "listening"
                    live.update(self._render())
            except KeyboardInterrupt:
                console.print("\nAda: Voice session ended.")
