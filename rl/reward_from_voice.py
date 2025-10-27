"""Reward extraction from voice tone and sentiment heuristics."""

from __future__ import annotations

from dataclasses import dataclass


import numpy as np


@dataclass
class VoiceReward:
    reward: float
    tone: str
    energy: float
    pitch: float
    voice_component: float


class VoiceRewardAnalyzer:
    def __init__(self, positive_threshold: float = 0.02, negative_threshold: float = 0.12) -> None:
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

    def analyze(self, audio: np.ndarray, sample_rate: int) -> VoiceReward:
        audio = audio.astype(np.float32)
        energy = float(np.mean(np.abs(audio)))
        pitch = self._estimate_pitch(audio, sample_rate)

        voice_reward = 0.0
        if energy < self.positive_threshold:
            tone = "calm"
            voice_reward = 1.0
        elif energy > self.negative_threshold:
            tone = "frustrated"
            voice_reward = -1.0
        else:
            tone = "neutral"

        if pitch > 260:  # elevated pitch often indicates excitement/stress
            tone = "excited"
            voice_reward = max(voice_reward, 0.5)
        elif pitch < 120 and energy > self.negative_threshold:
            tone = "firm"
            voice_reward = min(voice_reward, -0.5)

        return VoiceReward(
            reward=voice_reward,
            tone=tone,
            energy=energy,
            pitch=pitch,
            voice_component=voice_reward,
        )

    def _estimate_pitch(self, audio: np.ndarray, sample_rate: int) -> float:
        if len(audio) < sample_rate // 10:
            return 0.0
        audio = audio - np.mean(audio)
        corr = np.correlate(audio, audio, mode="full")
        corr = corr[len(corr) // 2 :]
        d = np.diff(corr)
        start = np.nonzero(d > 0)[0]
        if len(start) == 0:
            return 0.0
        peak = np.argmax(corr[start[0] :]) + start[0]
        if peak == 0:
            return 0.0
        frequency = sample_rate / peak
        return float(frequency)
