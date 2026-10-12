"""Walk the model folders and read each `.safetensors` header.

Read-only: nothing is written or moved."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from wasifyai_art_contracts import ModelKind, ScanProblem

from wasifyai_art_api.safetensors import Header, InvalidSafetensors, read_header

SUPPORTED_SUFFIX = ".safetensors"


@dataclass(frozen=True)
class ScannedFile:
    rel_path: str  # relative to the models folder, '/' separated
    folder: str
    kind: ModelKind
    size_bytes: int
    mtime: float
    header: Header


@dataclass
class ScanOutcome:
    found: list[ScannedFile]
    skipped_unsupported: list[str]
    problems: list[ScanProblem]


def load_folder_map(path: Path) -> dict[str, ModelKind]:
    data = yaml.safe_load(path.read_text()) or {}
    return dict(data.get("folders", {}))


def scan_models_dir(root: Path, folders: dict[str, ModelKind]) -> ScanOutcome:
    if not root.is_dir():
        raise FileNotFoundError(f"models folder not found: {root}")
    out = ScanOutcome([], [], [])
    for folder, kind in folders.items():
        base = root / folder
        if not base.is_dir():
            continue
        # followlinks=False: a symlinked folder is not entered (no loops, no escaping the root)
        for dirpath, _dirs, files in os.walk(base, followlinks=False):
            for name in sorted(files):
                path = Path(dirpath) / name
                rel = path.relative_to(root).as_posix()
                if name.startswith("."):
                    continue
                if not name.lower().endswith(SUPPORTED_SUFFIX):
                    out.skipped_unsupported.append(rel)
                    continue
                try:
                    st = path.stat()
                    header = read_header(path)
                except (InvalidSafetensors, OSError) as e:
                    out.problems.append(ScanProblem(rel_path=rel, reason=str(e)))
                    continue
                out.found.append(ScannedFile(rel, folder, kind, st.st_size, st.st_mtime, header))
    out.found.sort(key=lambda f: f.rel_path)
    out.skipped_unsupported.sort()
    return out
