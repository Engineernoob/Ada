"""Utilities for loading Ada configuration settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


@lru_cache(maxsize=1)
def load_settings() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_setting(*keys: str, default: Any | None = None) -> Any:
    settings = load_settings()
    node: Any = settings
    for key in keys:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return default
    return node
