from pathlib import Path

import yaml
from pydantic import BaseModel
from wasifyai_art_contracts import FamilyInfo


class GenerationDefaults(BaseModel):
    steps: int
    edit_steps: int | None = None
    sizes: dict[str, tuple[int, int]]


class FamilyConfig(BaseModel):
    id: str
    title: str
    match: list[str] = []
    generation: GenerationDefaults | None = None


def load_families(path: Path) -> list[FamilyConfig]:
    data = yaml.safe_load(path.read_text()) or {}
    return [FamilyConfig.model_validate(f) for f in data.get("families", [])]


def to_info(families: list[FamilyConfig]) -> list[FamilyInfo]:
    return [FamilyInfo(id=f.id, title=f.title) for f in families]


def guess_family(families: list[FamilyConfig], *sources: str) -> str | None:
    """A suggestion from metadata values and the file name; the first family with a match wins."""
    haystack = " ".join(s.lower() for s in sources if s)
    for f in families:
        if any(m.lower() in haystack for m in f.match):
            return f.id
    return None
