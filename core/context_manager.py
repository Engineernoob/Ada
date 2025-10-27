"""Context aggregation for Ada."""

from __future__ import annotations

from dataclasses import dataclass, field

from .memory import ConversationStore


@dataclass
class ContextManager:
    store: ConversationStore = field(default_factory=ConversationStore)
    max_history: int = 5

    def build_prompt(self, user_input: str) -> str:
        history = self.store.as_context(limit=self.max_history)
        if not history:
            return user_input
        return f"{history}\nYou: {user_input}"

    def remember(self, user_input: str, ada_response: str, reward: float) -> int:
        return self.store.log_interaction(user_input, ada_response, reward)
