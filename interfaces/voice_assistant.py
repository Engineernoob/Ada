"""Voice-enabled Ada assistant integrating speech and reinforcement learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from core import ContextManager, ReasoningEngine, get_setting
from core.event_loop import VoiceInteractionLoop
from interfaces.speech_input import SpeechInput
from interfaces.voice_output import VoiceOutput
from rl import (
    AdaAgent,
    DialogueEnvironment,
    ExperienceBuffer,
    VoiceReward,
    VoiceRewardAnalyzer,
)


@dataclass
class VoiceSession:
    context_manager: ContextManager
    engine: ReasoningEngine
    environment: DialogueEnvironment
    agent: AdaAgent
    voice_reward_analyzer: VoiceRewardAnalyzer
    speaker: VoiceOutput
    reward_scale: float = 0.5
    sample_rate: int = 16_000
    cumulative_reward: float = 0.0
    interactions: int = 0

    def process(
        self, transcript: str, audio_path: Path, audio_data: np.ndarray
    ) -> Tuple[str, VoiceReward]:
        state_vector = self.environment.encode(self.environment.state)
        prompt = self.context_manager.build_prompt(transcript)
        generation = self.engine.generate_with_metrics(prompt)
        ada_response = generation.text
        next_state = self.environment.observe(transcript, ada_response)
        next_state_vector = self.environment.encode(next_state)

        base_reward = self.engine.reward(transcript, ada_response)
        voice_reward = self.voice_reward_analyzer.analyze(audio_data, self.sample_rate)
        total_reward = base_reward + self.reward_scale * voice_reward.voice_component
        voice_reward = VoiceReward(
            reward=total_reward,
            tone=voice_reward.tone,
            energy=voice_reward.energy,
            pitch=voice_reward.pitch,
            voice_component=voice_reward.voice_component,
        )

        record_id = self.context_manager.remember(
            transcript, ada_response, total_reward
        )
        self.context_manager.store.update_reward(record_id, total_reward)

        self.agent.update_policy(
            state_vector, generation.action_index, total_reward, next_state_vector
        )
        _ = self.agent.train_on_batch()
        self.agent.save()

        self.interactions += 1
        self.cumulative_reward += total_reward

        return ada_response, voice_reward


def main() -> None:
    context = ContextManager()
    engine = ReasoningEngine()
    memory = ExperienceBuffer()
    environment = DialogueEnvironment()
    agent = AdaAgent(
        model=engine.model,
        memory=memory,
        action_space=engine.action_space,
        checkpoint_path=engine.checkpoint_path,
    )
    analyzer = VoiceRewardAnalyzer()
    speaker = VoiceOutput()
    reward_scale = float(get_setting("reinforcement", "reward_scale", default=0.5))
    session = VoiceSession(
        context_manager=context,
        engine=engine,
        environment=environment,
        agent=agent,
        voice_reward_analyzer=analyzer,
        speaker=speaker,
        reward_scale=reward_scale,
    )

    speech_input = SpeechInput()
    loop = VoiceInteractionLoop()

    def handler(text: str, audio_path: Path, audio_data: np.ndarray):
        response, voice_reward = session.process(text, audio_path, audio_data)
        return response, voice_reward

    loop.run(speech_input.phrases(), handler, speaker.speak)


if __name__ == "__main__":
    main()
