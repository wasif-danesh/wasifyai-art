"""Training data: datasets and their images, training runs, and the preset configuration.

Every query is scoped by tenant, product and owner (CLAUDE.md rule 4)."""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

import yaml
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from wasifyai_art_api import PRODUCT_ID


class FamilyTraining(BaseModel):
    engine: str
    model: str
    min_images: int = 5
    max_images: int = 100
    seconds_per_step: float = 7.5
    defaults: dict[str, Any]
    presets: dict[str, dict[str, Any]]


def load_training(path: Path) -> dict[str, FamilyTraining]:
    data = yaml.safe_load(path.read_text()) or {}
    return {k: FamilyTraining.model_validate(v) for k, v in data.get("families", {}).items()}


@dataclass
class DatasetRecord:
    id: str
    tenant_id: str
    owner_id: str
    name: str
    trigger_word: str
    created_at: str = ""


@dataclass
class ImageRecord:
    id: str
    dataset_id: str
    key: str
    thumb_key: str
    width: int
    height: int
    caption: str
    position: int = 0


@dataclass
class RunRecord:
    job_id: str
    dataset_id: str | None
    tenant_id: str
    owner_id: str
    name: str
    trigger_word: str
    family: str
    preset: str
    params: dict[str, Any]
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""


class TrainingStore(Protocol):
    async def create_dataset(
        self, tenant_id: str, owner_id: str, name: str, trigger: str
    ) -> DatasetRecord: ...

    async def list_datasets(self, tenant_id: str, owner_id: str) -> list[DatasetRecord]: ...

    async def get_dataset(
        self, tenant_id: str, owner_id: str, dataset_id: str
    ) -> DatasetRecord | None: ...

    async def delete_dataset(
        self, tenant_id: str, owner_id: str, dataset_id: str
    ) -> list[ImageRecord] | None:
        """Remove the dataset and return its images so the caller can delete the files."""
        ...

    async def add_image(self, dataset_id: str, rec: ImageRecord) -> None: ...

    async def list_images(self, dataset_id: str) -> list[ImageRecord]: ...

    async def get_image(self, dataset_id: str, image_id: str) -> ImageRecord | None: ...

    async def set_caption(self, dataset_id: str, image_id: str, caption: str) -> None: ...

    async def delete_image(self, dataset_id: str, image_id: str) -> None: ...

    async def create_run(self, rec: RunRecord) -> None: ...

    async def get_run(self, tenant_id: str, owner_id: str, job_id: str) -> RunRecord | None: ...

    async def get_run_any(self, job_id: str) -> RunRecord | None: ...

    async def list_runs(self, tenant_id: str, owner_id: str) -> list[RunRecord]: ...

    async def set_checkpoints(self, job_id: str, checkpoints: list[dict[str, Any]]) -> None: ...


def _now() -> str:
    return datetime.now(UTC).isoformat()


class InMemoryTrainingStore:
    def __init__(self) -> None:
        self.datasets: dict[str, DatasetRecord] = {}
        self.images: dict[str, list[ImageRecord]] = {}
        self.runs: dict[str, RunRecord] = {}

    async def create_dataset(
        self, tenant_id: str, owner_id: str, name: str, trigger: str
    ) -> DatasetRecord:
        rec = DatasetRecord(str(uuid4()), tenant_id, owner_id, name, trigger, _now())
        self.datasets[rec.id] = rec
        self.images[rec.id] = []
        return rec

    async def list_datasets(self, tenant_id: str, owner_id: str) -> list[DatasetRecord]:
        mine = [
            d for d in self.datasets.values() if (d.tenant_id, d.owner_id) == (tenant_id, owner_id)
        ]
        return sorted(mine, key=lambda d: d.created_at, reverse=True)

    async def get_dataset(
        self, tenant_id: str, owner_id: str, dataset_id: str
    ) -> DatasetRecord | None:
        d = self.datasets.get(dataset_id)
        return d if d and (d.tenant_id, d.owner_id) == (tenant_id, owner_id) else None

    async def delete_dataset(
        self, tenant_id: str, owner_id: str, dataset_id: str
    ) -> list[ImageRecord] | None:
        if await self.get_dataset(tenant_id, owner_id, dataset_id) is None:
            return None
        del self.datasets[dataset_id]
        for r in self.runs.values():
            if r.dataset_id == dataset_id:
                r.dataset_id = None
        return self.images.pop(dataset_id, [])

    async def add_image(self, dataset_id: str, rec: ImageRecord) -> None:
        rec.position = len(self.images[dataset_id])
        self.images[dataset_id].append(rec)

    async def list_images(self, dataset_id: str) -> list[ImageRecord]:
        return list(self.images.get(dataset_id, []))

    async def get_image(self, dataset_id: str, image_id: str) -> ImageRecord | None:
        return next((i for i in self.images.get(dataset_id, []) if i.id == image_id), None)

    async def set_caption(self, dataset_id: str, image_id: str, caption: str) -> None:
        if i := await self.get_image(dataset_id, image_id):
            i.caption = caption

    async def delete_image(self, dataset_id: str, image_id: str) -> None:
        self.images[dataset_id] = [i for i in self.images.get(dataset_id, []) if i.id != image_id]

    async def create_run(self, rec: RunRecord) -> None:
        rec.created_at = rec.created_at or _now()
        self.runs[rec.job_id] = rec

    async def get_run(self, tenant_id: str, owner_id: str, job_id: str) -> RunRecord | None:
        r = self.runs.get(job_id)
        return r if r and (r.tenant_id, r.owner_id) == (tenant_id, owner_id) else None

    async def get_run_any(self, job_id: str) -> RunRecord | None:
        return self.runs.get(job_id)

    async def list_runs(self, tenant_id: str, owner_id: str) -> list[RunRecord]:
        mine = [r for r in self.runs.values() if (r.tenant_id, r.owner_id) == (tenant_id, owner_id)]
        return sorted(mine, key=lambda r: r.created_at, reverse=True)

    async def set_checkpoints(self, job_id: str, checkpoints: list[dict[str, Any]]) -> None:
        if r := self.runs.get(job_id):
            r.checkpoints = checkpoints


def _uuid(v: str) -> bool:
    try:
        UUID(v)
        return True
    except ValueError:
        return False


class PostgresTrainingStore:
    def __init__(self, engine: AsyncEngine):
        self._e = engine

    async def create_dataset(
        self, tenant_id: str, owner_id: str, name: str, trigger: str
    ) -> DatasetRecord:
        async with self._e.begin() as c:
            r = (
                await c.execute(
                    text(
                        """INSERT INTO datasets
                            (tenant_id, product_id, owner_id, name, trigger_word)
                        VALUES (:t, :p, :o, :n, :w) RETURNING id::text, created_at"""
                    ),
                    {"t": tenant_id, "p": PRODUCT_ID, "o": owner_id, "n": name, "w": trigger},
                )
            ).one()
        return DatasetRecord(r[0], tenant_id, owner_id, name, trigger, r[1].isoformat())

    async def list_datasets(self, tenant_id: str, owner_id: str) -> list[DatasetRecord]:
        async with self._e.connect() as c:
            rows = (
                await c.execute(
                    text(
                        """SELECT id::text, name, trigger_word, created_at FROM datasets
                        WHERE tenant_id = :t AND product_id = :p AND owner_id = :o
                        ORDER BY created_at DESC"""
                    ),
                    {"t": tenant_id, "p": PRODUCT_ID, "o": owner_id},
                )
            ).all()
        return [
            DatasetRecord(r[0], tenant_id, owner_id, r[1], r[2], r[3].isoformat()) for r in rows
        ]

    async def get_dataset(
        self, tenant_id: str, owner_id: str, dataset_id: str
    ) -> DatasetRecord | None:
        if not _uuid(dataset_id):
            return None
        async with self._e.connect() as c:
            r = (
                await c.execute(
                    text(
                        """SELECT id::text, name, trigger_word, created_at FROM datasets
                        WHERE id = CAST(:i AS uuid) AND tenant_id = :t AND product_id = :p
                          AND owner_id = :o"""
                    ),
                    {"i": dataset_id, "t": tenant_id, "p": PRODUCT_ID, "o": owner_id},
                )
            ).first()
        return DatasetRecord(r[0], tenant_id, owner_id, r[1], r[2], r[3].isoformat()) if r else None

    async def delete_dataset(
        self, tenant_id: str, owner_id: str, dataset_id: str
    ) -> list[ImageRecord] | None:
        if await self.get_dataset(tenant_id, owner_id, dataset_id) is None:
            return None
        images = await self.list_images(dataset_id)
        async with self._e.begin() as c:
            await c.execute(
                text("DELETE FROM datasets WHERE id = CAST(:i AS uuid)"), {"i": dataset_id}
            )
        return images

    async def add_image(self, dataset_id: str, rec: ImageRecord) -> None:
        async with self._e.begin() as c:
            await c.execute(
                text(
                    """INSERT INTO dataset_images
                        (id, dataset_id, key, thumb_key, width, height, caption, position)
                    VALUES (CAST(:i AS uuid), CAST(:d AS uuid), :k, :tk, :w, :h, :c,
                            (SELECT COALESCE(MAX(position) + 1, 0) FROM dataset_images
                             WHERE dataset_id = CAST(:d AS uuid)))"""
                ),
                {"i": rec.id, "d": dataset_id, "k": rec.key, "tk": rec.thumb_key, "w": rec.width,
                 "h": rec.height, "c": rec.caption},
            )  # fmt: skip

    @staticmethod
    def _img(r: Any) -> ImageRecord:
        return ImageRecord(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7])

    _ICOLS = "id::text, dataset_id::text, key, thumb_key, width, height, caption, position"

    async def list_images(self, dataset_id: str) -> list[ImageRecord]:
        async with self._e.connect() as c:
            rows = (
                await c.execute(
                    text(
                        f"SELECT {self._ICOLS} FROM dataset_images "
                        "WHERE dataset_id = CAST(:d AS uuid) ORDER BY position, created_at"
                    ),
                    {"d": dataset_id},
                )
            ).all()
        return [self._img(r) for r in rows]

    async def get_image(self, dataset_id: str, image_id: str) -> ImageRecord | None:
        if not _uuid(image_id):
            return None
        async with self._e.connect() as c:
            r = (
                await c.execute(
                    text(
                        f"SELECT {self._ICOLS} FROM dataset_images WHERE id = CAST(:i AS uuid) "
                        "AND dataset_id = CAST(:d AS uuid)"
                    ),
                    {"i": image_id, "d": dataset_id},
                )
            ).first()
        return self._img(r) if r else None

    async def set_caption(self, dataset_id: str, image_id: str, caption: str) -> None:
        async with self._e.begin() as c:
            await c.execute(
                text(
                    "UPDATE dataset_images SET caption = :c WHERE id = CAST(:i AS uuid) "
                    "AND dataset_id = CAST(:d AS uuid)"
                ),
                {"c": caption, "i": image_id, "d": dataset_id},
            )

    async def delete_image(self, dataset_id: str, image_id: str) -> None:
        async with self._e.begin() as c:
            await c.execute(
                text(
                    "DELETE FROM dataset_images WHERE id = CAST(:i AS uuid) "
                    "AND dataset_id = CAST(:d AS uuid)"
                ),
                {"i": image_id, "d": dataset_id},
            )

    async def create_run(self, rec: RunRecord) -> None:
        async with self._e.begin() as c:
            await c.execute(
                text(
                    """INSERT INTO training_runs
                        (job_id, dataset_id, tenant_id, product_id, owner_id, name, trigger_word,
                         family, preset, params)
                    VALUES (CAST(:j AS uuid), CAST(:d AS uuid), :t, :p, :o, :n, :w, :f, :pr,
                            CAST(:pa AS jsonb))"""
                ),
                {"j": rec.job_id, "d": rec.dataset_id, "t": rec.tenant_id, "p": PRODUCT_ID,
                 "o": rec.owner_id, "n": rec.name, "w": rec.trigger_word, "f": rec.family,
                 "pr": rec.preset, "pa": json.dumps(rec.params)},
            )  # fmt: skip

    _RCOLS = (
        "job_id::text, dataset_id::text, tenant_id, owner_id, name, trigger_word, family, preset, "
        "params, checkpoints, created_at"
    )

    @staticmethod
    def _run(r: Any) -> RunRecord:
        return RunRecord(
            r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10].isoformat()
        )

    async def get_run(self, tenant_id: str, owner_id: str, job_id: str) -> RunRecord | None:
        if not _uuid(job_id):
            return None
        async with self._e.connect() as c:
            r = (
                await c.execute(
                    text(
                        f"""SELECT {self._RCOLS} FROM training_runs WHERE job_id = CAST(:j AS uuid)
                        AND tenant_id = :t AND product_id = :p AND owner_id = :o"""
                    ),
                    {"j": job_id, "t": tenant_id, "p": PRODUCT_ID, "o": owner_id},
                )
            ).first()
        return self._run(r) if r else None

    async def get_run_any(self, job_id: str) -> RunRecord | None:
        if not _uuid(job_id):
            return None
        async with self._e.connect() as c:
            r = (
                await c.execute(
                    text(
                        f"SELECT {self._RCOLS} FROM training_runs WHERE job_id = CAST(:j AS uuid)"
                    ),
                    {"j": job_id},
                )
            ).first()
        return self._run(r) if r else None

    async def list_runs(self, tenant_id: str, owner_id: str) -> list[RunRecord]:
        async with self._e.connect() as c:
            rows = (
                await c.execute(
                    text(
                        f"""SELECT {self._RCOLS} FROM training_runs WHERE tenant_id = :t
                        AND product_id = :p AND owner_id = :o ORDER BY created_at DESC"""
                    ),
                    {"t": tenant_id, "p": PRODUCT_ID, "o": owner_id},
                )
            ).all()
        return [self._run(r) for r in rows]

    async def set_checkpoints(self, job_id: str, checkpoints: list[dict[str, Any]]) -> None:
        async with self._e.begin() as c:
            await c.execute(
                text(
                    "UPDATE training_runs SET checkpoints = CAST(:c AS jsonb) "
                    "WHERE job_id = CAST(:j AS uuid)"
                ),
                {"c": json.dumps(checkpoints), "j": job_id},
            )
