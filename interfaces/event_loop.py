"""Asynchronous event loop placeholder for Ada interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class EventLoop:
    prompt: str = "You: "

    def run(self, handler: Callable[[str], str]) -> None:
        try:
            while True:
                user_input = input(self.prompt).strip()
                if user_input.lower() in {"exit", "quit"}:
                    print("Ada: Until next time.")
                    break
                response = handler(user_input)
                print(f"Ada: {response}")
        except KeyboardInterrupt:
            print("\nAda: Session ended.")
