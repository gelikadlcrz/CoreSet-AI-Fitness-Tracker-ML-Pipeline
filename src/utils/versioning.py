"""
versioning.py — small helpers for versioned ST-GCN training outputs.

The training script writes each completed run to its own folder, for example:
    checkpoint/stgcn_v001/
    checkpoint/stgcn_v002/

This prevents a new training run from overwriting a previous trained model.
Evaluation/export scripts can pass "latest" to automatically use the newest
saved checkpoint.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional


_VERSION_RE_TEMPLATE = r"^{prefix}_v(\d+)$"


def create_versioned_run_dir(checkpoint_root: str | Path, prefix: str = "stgcn") -> Path:
    """Create and return the next run directory, e.g. checkpoint/stgcn_v003."""
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(_VERSION_RE_TEMPLATE.format(prefix=re.escape(prefix)))
    existing_versions = []
    for child in root.iterdir():
        if child.is_dir():
            match = pattern.match(child.name)
            if match:
                existing_versions.append(int(match.group(1)))

    next_version = (max(existing_versions) + 1) if existing_versions else 1
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
    """Write a tiny text pointer to the newest checkpoint path."""
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    pointer = root / f"latest_{prefix}_checkpoint.txt"
    pointer.write_text(str(Path(checkpoint_path).as_posix()), encoding="utf-8")
    return pointer


def write_latest_run_pointer(
    checkpoint_root: str | Path,
    run_dir: str | Path,
    prefix: str = "stgcn",
) -> Path:
    """Write a tiny text pointer to the newest run folder."""
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    pointer = root / f"latest_{prefix}_run.txt"
    pointer.write_text(str(Path(run_dir).as_posix()), encoding="utf-8")
    return pointer


def _existing(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.exists() and p.is_file()]


def resolve_checkpoint_path(
    checkpoint_path: str | Path = "latest",
    checkpoint_root: str | Path = "checkpoint",
    prefix: str = "stgcn",
) -> Path:
    """
    Resolve a checkpoint path.

    Accepted forms:
      - "latest" or "auto": use the newest versioned checkpoint.
      - a direct .pth path: use it if it exists.
      - the older default checkpoint/best_stgcn_model.pth: use it if present,
        otherwise fall back to the newest versioned checkpoint.
    """
    root = Path(checkpoint_root)
    raw = str(checkpoint_path).strip() if checkpoint_path is not None else "latest"
    requested = Path(raw)

    if raw.lower() not in {"", "latest", "auto"} and requested.exists():
        return requested

    # If the user gave a specific missing path, fail clearly.  The only
    # exception is the legacy default, which can safely fall back to latest.
    legacy_default = requested.as_posix().replace("\\", "/") == f"{checkpoint_root}/best_stgcn_model.pth"
    if raw.lower() not in {"", "latest", "auto"} and not legacy_default:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    pointer = root / f"latest_{prefix}_checkpoint.txt"
    if pointer.exists():
        candidate = Path(pointer.read_text(encoding="utf-8").strip())
        if candidate.exists():
            return candidate

    candidates = []
    candidates.extend(root.glob(f"{prefix}_v*/best_{prefix}_model.pth"))
    candidates.extend(root.glob(f"{prefix}_v*/best_stgcn_model.pth"))
    candidates.extend(root.glob(f"{prefix}_v*/epoch_*.pth"))
    candidates.extend([root / "best_stgcn_model.pth"])

    existing = _existing(candidates)
    if existing:
        return max(existing, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(
        f"No {prefix.upper()} checkpoint found under '{root}'. "
        f"Train the model first or pass --checkpoint path/to/model.pth."
    )


def make_unique_file_path(path: str | Path) -> Path:
    """Return a non-existing file path by appending _v02, _v03, ... when needed."""
    path = Path(path)
    if not path.exists():
        return path
    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem}_v{counter:02d}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1
