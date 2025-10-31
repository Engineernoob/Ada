"""Tone detection and response modulation for Ada."""

from __future__ import annotations

import re


TONE_EMOJIS = {
    "happy": "😊",
    "sad": "💙",
    "angry": "😠",
    "excited": "🤩",
    "neutral": "✨",
}


def detect_tone(text: str) -> str:
    """Infer a coarse emotional tone from user input."""

    lowered = text.lower()
    if re.search(r"\b(thank|love|great|amazing|appreciate)\b", lowered):
        return "happy"
    if re.search(r"\b(sad|tired|bad|upset|depressed)\b", lowered):
        return "sad"
    if re.search(r"\b(angry|mad|frustrated|annoyed)\b", lowered):
        return "angry"
    if re.search(r"\b(excited|wow|awesome|incredible)\b", lowered):
        return "excited"
    return "neutral"


def adjust_response_tone(response: str, tone: str) -> str:
    """Prefix the response with a tone marker."""

    marker = TONE_EMOJIS.get(tone, "")
    return f"{marker} {response}".strip()
