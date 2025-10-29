"""Checkpoint management helpers for Ada's optimizer."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from core.settings import get_setting

from . import ROOT_PATH

LOGGER = logging.getLogger("ada.optimizer.checkpoints")
CHECKPOINT_DIR = ROOT_PATH / "storage" / "checkpoints"
METADATA_PATH = ROOT_PATH / "storage" / "optimizer" / "checkpoints.json"


@dataclass(slots=True)
class CheckpointMetadata:
    """Metadata describing a stored checkpoint."""

    checkpoint_id: str
    path: Path
    created_at: datetime
    reward_delta: float
    params: dict[str, Any]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        payload["created_at"] = self.created_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CheckpointMetadata":
        return cls(
            checkpoint_id=payload["checkpoint_id"],
            path=Path(payload["path"]),
            created_at=datetime.fromisoformat(payload["created_at"]),
            reward_delta=float(payload["reward_delta"]),
            params=payload.get("params", {}),
            notes=payload.get("notes", ""),
        )


class CheckpointManager:
    """Stores and prunes optimizer checkpoints."""

    def __init__(
        self,
        directory: Path | None = None,
        metadata_path: Path | None = None,
        keep_top: int = 3,
        restore_callback: Callable[[Path], None] | None = None,
    ) -> None:
        self.directory = Path(directory) if directory else CHECKPOINT_DIR
        self.metadata_path = Path(metadata_path) if metadata_path else METADATA_PATH
        self.keep_top = max(1, keep_top)
        self.restore_callback = restore_callback
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._configure_logger()

    # ------------------------------------------------------------------
    def promote(
        self,
        *,
        reward_delta: float,
        params: dict[str, Any],
        source_path: Path | None = None,
        notes: str = "",
    ) -> CheckpointMetadata:
        """Persist a new checkpoint and update metadata."""

        timestamp = datetime.now(timezone.utc)
        checkpoint_id = timestamp.strftime("ckpt-%Y%m%d-%H%M%S")
        checkpoint_path = self.directory / f"{checkpoint_id}.pt"
        if source_path and source_path.exists():
            shutil.copy2(source_path, checkpoint_path)
        else:
            checkpoint_path.write_text(json.dumps({"metadata_only": True}, indent=2))
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            path=checkpoint_path,
            created_at=timestamp,
            reward_delta=reward_delta,
            params=params,
            notes=notes,
        )
        entries = self._load_entries()
        entries.append(metadata)
        entries.sort(key=lambda item: item.reward_delta, reverse=True)
        if len(entries) > self.keep_top:
            for stale in entries[self.keep_top :]:
                if stale.path.exists():
                    try:
                        stale.path.unlink()
                    except OSError:
                        LOGGER.warning("Unable to remove stale checkpoint %s", stale.path)
            entries = entries[: self.keep_top]
        self._write_entries(entries)
        LOGGER.info(
            "Checkpoint promoted %s | reward_delta=%.3f | path=%s",
            checkpoint_id,
            reward_delta,
            checkpoint_path,
        )
        return metadata

    def list_checkpoints(self) -> list[CheckpointMetadata]:
        """Return all stored checkpoint metadata sorted by reward delta."""

        entries = self._load_entries()
        entries.sort(key=lambda item: item.reward_delta, reverse=True)
        return entries

    def rollback(self, checkpoint_id: str) -> CheckpointMetadata | None:
        """Restore the given checkpoint to the active model path."""

        for entry in self._load_entries():
            if entry.checkpoint_id == checkpoint_id:
                self._restore(entry)
                return entry
        return None

    # ------------------------------------------------------------------
    def _restore(self, metadata: CheckpointMetadata) -> None:
        active_path_str = get_setting("paths", {}).get("checkpoints")
        if active_path_str:
            active_path = Path(active_path_str)
            active_path.parent.mkdir(parents=True, exist_ok=True)
            if metadata.path.exists():
                shutil.copy2(metadata.path, active_path)
            else:
                active_path.write_text(json.dumps({"restored": metadata.checkpoint_id}, indent=2))
        if self.restore_callback:
            self.restore_callback(metadata.path)
        LOGGER.info("Rollback applied for %s", metadata.checkpoint_id)

    def _configure_logger(self) -> None:
        LOGGER.setLevel(logging.INFO)
        if not LOGGER.handlers:
            handler = logging.FileHandler(ROOT_PATH / "storage" / "logs" / "optimizer.log", encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            )
            LOGGER.addHandler(handler)
        LOGGER.propagate = False

    def _load_entries(self) -> list[CheckpointMetadata]:
        if not self.metadata_path.exists():
            return []
        payload = json.loads(self.metadata_path.read_text())
        return [CheckpointMetadata.from_dict(item) for item in payload]

    def _write_entries(self, entries: list[CheckpointMetadata]) -> None:
        data = [entry.to_dict() for entry in entries]
        self.metadata_path.write_text(json.dumps(data, indent=2))
