"""
versioning.py — helpers for clean versioned ST-GCN outputs.

Expected checkpoint structure:

    checkpoint/
      latest_stgcn_checkpoint.txt
      latest_stgcn_run.txt

      stgcn_v001/
        final_stgcn_model.pth
        training_history.csv
        config_snapshot.yaml
        run_metadata.json

      stgcn_v002/
        final_stgcn_model.pth
        training_history.csv
        config_snapshot.yaml
        run_metadata.json

Rules:
- Each training run gets its own version folder.
- Each version folder contains only one .pth model file.
- "latest" resolves only to final_stgcn_model.pth.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


_VERSION_RE_TEMPLATE = r"^{prefix}_v(\d+)$"
FINAL_MODEL_NAME = "final_stgcn_model.pth"


def create_versioned_run_dir(
    checkpoint_root: str | Path,
    prefix: str = "stgcn",
) -> Path:
    """
    Create and return the next versioned run directory.

    Example:
        checkpoint/stgcn_v001
        checkpoint/stgcn_v002
        checkpoint/stgcn_v003
    """
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(
        _VERSION_RE_TEMPLATE.format(prefix=re.escape(prefix))
    )

    existing_versions: list[int] = []

    for child in root.iterdir():
        if not child.is_dir():
            continue

        match = pattern.match(child.name)

        if match:
            existing_versions.append(int(match.group(1)))

    next_version = max(existing_versions, default=0) + 1

    while True:
        run_dir = root / f"{prefix}_v{next_version:03d}"

        try:
            run_dir.mkdir(parents=False, exist_ok=False)
            return run_dir
        except FileExistsError:
            next_version += 1


def write_latest_checkpoint_pointer(
    checkpoint_root: str | Path,
    checkpoint_path: str | Path,
    prefix: str = "stgcn",
) -> Path:
    """
    Write a pointer file to the latest final checkpoint.

    Example pointer content:
        checkpoint/stgcn_v002/final_stgcn_model.pth
    """
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(checkpoint_path)

    pointer = root / f"latest_{prefix}_checkpoint.txt"
    pointer.write_text(
        checkpoint_path.as_posix(),
        encoding="utf-8",
    )

    return pointer


def write_latest_run_pointer(
    checkpoint_root: str | Path,
    run_dir: str | Path,
    prefix: str = "stgcn",
) -> Path:
    """
    Write a pointer file to the latest versioned run folder.

    Example pointer content:
        checkpoint/stgcn_v002
    """
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)

    run_dir = Path(run_dir)

    pointer = root / f"latest_{prefix}_run.txt"
    pointer.write_text(
        run_dir.as_posix(),
        encoding="utf-8",
    )

    return pointer


def _existing(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.exists() and p.is_file()]


def _version_number_from_path(path: Path, prefix: str = "stgcn") -> int:
    """
    Extract version number from:
        checkpoint/stgcn_v###/final_stgcn_model.pth
    """
    pattern = re.compile(
        _VERSION_RE_TEMPLATE.format(prefix=re.escape(prefix))
    )

    match = pattern.match(path.parent.name)

    if not match:
        return -1

    return int(match.group(1))


def _is_final_model_path(path: Path) -> bool:
    return path.name == FINAL_MODEL_NAME


def _resolve_pointer_path(pointer_text: str, checkpoint_root: Path) -> Path | None:
    """
    Resolve a path stored in latest_stgcn_checkpoint.txt.

    Supports:
        checkpoint/stgcn_v001/final_stgcn_model.pth
        stgcn_v001/final_stgcn_model.pth
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
    checkpoint_root: str | Path = "checkpoint",
    prefix: str = "stgcn",
) -> Path:
    """
    Resolve which checkpoint to evaluate/export.

    Accepted forms:
        latest
        auto
        checkpoint/stgcn_v001/final_stgcn_model.pth

    Automatic resolution only uses final_stgcn_model.pth.

    It does not fall back to old messy checkpoint names like:
        best_stgcn_model.pth
        best_stgcn_by_accuracy.pth
        best_stgcn_by_counting.pth
        score_epoch_*.pth
        accuracy_epoch_*.pth
        counting_epoch_*.pth
    """
    root = Path(checkpoint_root)
    raw = str(checkpoint_path).strip() if checkpoint_path is not None else "latest"

    if raw.lower() not in {"", "latest", "auto"}:
        requested = Path(raw)

        if requested.exists() and requested.is_file():
            return requested

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    pointer = root / f"latest_{prefix}_checkpoint.txt"

    if pointer.exists():
        pointer_candidate = _resolve_pointer_path(
            pointer.read_text(encoding="utf-8"),
            root,
        )

        if (
            pointer_candidate is not None
            and pointer_candidate.exists()
            and pointer_candidate.is_file()
            and _is_final_model_path(pointer_candidate)
        ):
            return pointer_candidate

    candidates = _existing(
        root.glob(f"{prefix}_v*/{FINAL_MODEL_NAME}")
    )

    if candidates:
        candidates.sort(
            key=lambda p: (
                _version_number_from_path(p, prefix),
                p.stat().st_mtime,
            ),
            reverse=True,
        )

        return candidates[0]

    raise FileNotFoundError(
        f"No final {prefix.upper()} checkpoint found under '{root}'. "
        f"Train the model first. Expected file pattern: "
        f"{root.as_posix()}/{prefix}_v###/{FINAL_MODEL_NAME}"
    )


def make_unique_file_path(path: str | Path) -> Path:
    """
    Return a non-existing file path by appending _v02, _v03, etc.

    Used for exported ONNX/TFLite files only.
    Training checkpoints should remain final_stgcn_model.pth inside their
    own version folder.
    """
    path = Path(path)

    if not path.exists():
        return path

    counter = 2

    while True:
        candidate = path.with_name(
            f"{path.stem}_v{counter:02d}{path.suffix}"
        )

        if not candidate.exists():
            return candidate

        counter += 1