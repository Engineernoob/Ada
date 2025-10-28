"""Utilities for loading Ada configuration settings."""

from __future__ import annotations

from ast import literal_eval
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

try:  # Optional dependency: PyYAML offers a full parser when available
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - exercised in minimal environments
    yaml = None


def _parse_value(raw: str) -> Any:
    """Parse a simple YAML scalar value.

    The project configuration only uses a small subset of YAML (quoted strings,
    numbers, booleans and inline lists).  ``literal_eval`` safely handles these
    cases, so we fall back to it when PyYAML isn't installed.
    """

    text = raw.strip()
    if not text:
        return ""

    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        return literal_eval(text)
    except (ValueError, SyntaxError):
        # Treat bare words as strings (mirrors YAML's behaviour for our usage).
        return text


def _parse_simple_yaml(content: str) -> Dict[str, Any]:
    """Very small YAML subset parser used when PyYAML isn't available."""

    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(0, root)]

    for raw_line in content.splitlines():
        # Strip comments and surrounding whitespace
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue

        indent = len(line) - len(line.lstrip(" "))
        key_part, _, value_part = line.lstrip().partition(":")
        key = key_part.strip()
        value = value_part.strip()

        while stack and indent < stack[-1][0]:
            stack.pop()

        current = stack[-1][1]
        if not value:
            new_node: Dict[str, Any] = {}
            current[key] = new_node
            stack.append((indent + 2, new_node))
        else:
            current[key] = _parse_value(value)

    return root


@lru_cache(maxsize=1)
def load_settings() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        content = handle.read()
    if yaml is not None:
        return yaml.safe_load(content)
    return _parse_simple_yaml(content)


def get_setting(*keys: str, default: Any | None = None) -> Any:
    settings = load_settings()
    node: Any = settings
    for key in keys:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return default
    return node
