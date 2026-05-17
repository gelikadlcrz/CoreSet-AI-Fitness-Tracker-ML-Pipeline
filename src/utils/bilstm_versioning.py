"""
bilstm_versioning.py — helpers for clean versioned BiLSTM outputs.

Expected checkpoint structure:

    saved_models/
      latest_bilstm_checkpoint.txt
      latest_bilstm_run.txt

      bilstm_v001/
        final_bilstm_model.pth
        training_history.csv
        config_snapshot.yaml
        run_metadata.json

      bilstm_v002/
        final_bilstm_model.pth
        training_history.csv
        config_snapshot.yaml
        run_metadata.json

Rules:
- Each training run gets its own version folder.
- Each version folder contains only one .pth model file.
- "latest" resolves only to final_bilstm_model.pth.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


_VERSION_RE = re.compile(r"^bilstm_v(\d+)$")
FINAL_MODEL_NAME = "final_bilstm_model.pth"


def create_versioned_run_dir(checkpoint_root: str | Path) -> Path:
    """
    Create and return the next versioned run directory.

    Example:
        saved_models/bilstm_v001
        saved_models/bilstm_v002
        saved_models/bilstm_v003
    """
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)

    existing_versions: list[int] = []

    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = _VERSION_RE.match(child.name)
        if match:
            existing_versions.append(int(match.group(1)))

    next_version = max(existing_versions, default=0) + 1

    while True:
        run_dir = root / f"bilstm_v{next_version:03d}"
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
            return run_dir
        except FileExistsError:
            next_version += 1


def write_latest_checkpoint_pointer(
    checkpoint_root: str | Path,
    checkpoint_path: str | Path,
) -> Path:
    """
    Write a pointer file to the latest final checkpoint.

    Example pointer content:
        saved_models/bilstm_v002/final_bilstm_model.pth
    """
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)

    pointer = root / "latest_bilstm_checkpoint.txt"
    pointer.write_text(Path(checkpoint_path).as_posix(), encoding="utf-8")

    return pointer


def write_latest_run_pointer(
    checkpoint_root: str | Path,
    run_dir: str | Path,
) -> Path:
    """
    Write a pointer file to the latest versioned run folder.

    Example pointer content:
        saved_models/bilstm_v002
    """
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)

    pointer = root / "latest_bilstm_run.txt"
    pointer.write_text(Path(run_dir).as_posix(), encoding="utf-8")

    return pointer


def _existing(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.exists() and p.is_file()]


def _version_number_from_path(path: Path) -> int:
    """
    Extract version number from:
        saved_models/bilstm_v###/final_bilstm_model.pth
    """
    match = _VERSION_RE.match(path.parent.name)
    return int(match.group(1)) if match else -1


def _resolve_pointer_path(pointer_text: str, checkpoint_root: Path) -> Path | None:
    """
    Resolve a path stored in latest_bilstm_checkpoint.txt.

    Supports:
        saved_models/bilstm_v001/final_bilstm_model.pth
        bilstm_v001/final_bilstm_model.pth
    """
    raw = pointer_text.strip()
    if not raw:
        return None

    candidate = Path(raw)
    if candidate.exists():
        return candidate

    candidate_from_root = checkpoint_root / candidate
    if candidate_from_root.exists():
        return candidate_from_root

    return None


def resolve_checkpoint_path(
    checkpoint_path: str | Path = "latest",
    checkpoint_root: str | Path = "saved_models",
) -> Path:
    """
    Resolve which BiLSTM checkpoint to evaluate/export.

    Accepted forms:
        latest
        auto
        saved_models/bilstm_v001/final_bilstm_model.pth

    Automatic resolution only uses final_bilstm_model.pth.

    It does not fall back to old checkpoint names like:
        best_bilstm.pth
        best_bilstm_baseline.pth
    """
    root = Path(checkpoint_root)
    raw = str(checkpoint_path).strip() if checkpoint_path is not None else "latest"

    if raw.lower() not in {"", "latest", "auto"}:
        requested = Path(raw)
        if requested.exists() and requested.is_file():
            return requested
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    pointer = root / "latest_bilstm_checkpoint.txt"

    if pointer.exists():
        pointer_candidate = _resolve_pointer_path(
            pointer.read_text(encoding="utf-8"), root
        )
        if (
            pointer_candidate is not None
            and pointer_candidate.exists()
            and pointer_candidate.is_file()
            and pointer_candidate.name == FINAL_MODEL_NAME
        ):
            return pointer_candidate

    candidates = _existing(root.glob(f"bilstm_v*/{FINAL_MODEL_NAME}"))

    if candidates:
        candidates.sort(
            key=lambda p: (_version_number_from_path(p), p.stat().st_mtime),
            reverse=True,
        )
        return candidates[0]

    raise FileNotFoundError(
        f"No final BiLSTM checkpoint found under '{root}'. "
        f"Train the model first. Expected file pattern: "
        f"{root.as_posix()}/bilstm_v###/{FINAL_MODEL_NAME}"
    )
