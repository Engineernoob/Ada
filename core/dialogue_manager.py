"""Conversation orchestration for Ada."""

from __future__ import annotations

from typing import Tuple

from .emotion_tone import adjust_response_tone, detect_tone
from .memory import add_memory, recall
from .session_state import get_context, save_turn


def build_conversation_context(prompt: str, limit: int = 5) -> str:
    """Combine recent session turns with semantic recall."""

    history = get_context(limit=limit)
    recalled = recall(prompt, limit=limit)
    unordered = [*history, *recalled]
    segments = [segment.strip() for segment in unordered if segment.strip()]
    return "\n".join(segments)


def process_conversation(prompt: str, model) -> Tuple[str, float, str]:
    """Generate a conversational response with tone adaptation."""

    context = build_conversation_context(prompt)
    tone = detect_tone(prompt)
    if context:
        input_text = f"{context}\nUser: {prompt}\nAda:"
    else:
        input_text = f"User: {prompt}\nAda:"

    response, confidence = _generate_with_confidence(model, input_text)
    adjusted = adjust_response_tone(response, tone)

    add_memory(prompt, adjusted)
    save_turn(prompt, adjusted)

    return adjusted, confidence, tone


def _generate_with_confidence(model, input_text: str) -> Tuple[str, float]:
    if hasattr(model, "generate_with_metrics"):
        result = model.generate_with_metrics(input_text)
        text = getattr(result, "text", "")
        confidence = float(getattr(result, "confidence", 0.0) or 0.0)
        return text, confidence

    if hasattr(model, "generate"):
        text = model.generate(input_text)
        confidence = getattr(model, "confidence", 0.0)
        try:
            return text, float(confidence)
        except (TypeError, ValueError):
            return text, 0.0

    raise AttributeError("Model must expose generate or generate_with_metrics method")
