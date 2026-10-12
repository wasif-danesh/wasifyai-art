"""Jobs and creations. Every query is scoped by tenant, product and owner (CLAUDE.md rule 4)."""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from wasifyai_art_contracts import CreationModelRef

from wasifyai_art_api import PRODUCT_ID

MAX_ATTEMPTS = 3


@dataclass
class JobRecord:
    id: str
    tenant_id: str
    owner_id: str
    status: str
    request: dict[str, Any]
    progress: float = 0.0
    error: str | None = None
    creation_id: str | None = None
    attempts: int = 0
    created_at: str = ""


@dataclass
class CreationRecord:
    id: str
    prompt: str
    params: dict[str, Any]
    provider_id: str
    image_key: str
    thumb_key: str
    width: int
    height: int
    created_at: str
    models: list[CreationModelRef] = field(default_factory=list)
    parent_id: str | None = None


class JobStore(Protocol):
    async def create_job(
        self, tenant_id: str, owner_id: str, request: dict[str, Any]
    ) -> JobRecord: ...

    async def get_job(self, tenant_id: str, owner_id: str, job_id: str) -> JobRecord | None: ...

    async def get_job_any(self, job_id: str) -> JobRecord | None: ...

    async def list_jobs(self, tenant_id: str, owner_id: str, limit: int) -> list[JobRecord]: ...

    async def cancel_queued(self, tenant_id: str, owner_id: str, job_id: str) -> JobRecord | None:
        """Cancel a job that has not started. None when it is not queued (or not yours)."""
        ...

    async def start_job(self, job_id: str) -> JobRecord | None: ...

    async def set_progress(self, job_id: str, progress: float) -> None: ...

    async def finish_job(
        self,
        job_id: str,
        status: str,
        error: str | None,
        creation_id: str | None,
        gpu_seconds: float,
    ) -> None: ...

    async def add_creation(self, tenant_id: str, owner_id: str, rec: CreationRecord) -> None: ...

    async def list_creations(
        self, tenant_id: str, owner_id: str, limit: int
    ) -> list[CreationRecord]: ...

    async def get_creation(
        self, tenant_id: str, owner_id: str, creation_id: str
    ) -> CreationRecord | None: ...

    async def delete_creation(
        self, tenant_id: str, owner_id: str, creation_id: str
    ) -> CreationRecord | None: ...


def _now() -> str:
    return datetime.now(UTC).isoformat()


class InMemoryJobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, tuple[str, JobRecord]] = {}
        self.creations: list[tuple[str, str, CreationRecord]] = []

    async def create_job(self, tenant_id: str, owner_id: str, request: dict[str, Any]) -> JobRecord:
        rec = JobRecord(str(uuid4()), tenant_id, owner_id, "queued", request, created_at=_now())
        self.jobs[rec.id] = (tenant_id, rec)
        return rec

    async def get_job(self, tenant_id: str, owner_id: str, job_id: str) -> JobRecord | None:
        row = self.jobs.get(job_id)
        if row and row[1].tenant_id == tenant_id and row[1].owner_id == owner_id:
            return row[1]
        return None

    async def get_job_any(self, job_id: str) -> JobRecord | None:
        row = self.jobs.get(job_id)
        return row[1] if row else None

    async def list_jobs(self, tenant_id: str, owner_id: str, limit: int) -> list[JobRecord]:
        mine = [
            r for _t, r in self.jobs.values() if (r.tenant_id, r.owner_id) == (tenant_id, owner_id)
        ]
        return sorted(mine, key=lambda r: r.created_at, reverse=True)[:limit]

    async def cancel_queued(self, tenant_id: str, owner_id: str, job_id: str) -> JobRecord | None:
        rec = await self.get_job(tenant_id, owner_id, job_id)
        if rec is None or rec.status != "queued":
            return None
        rec.status = "canceled"
        return rec

    async def start_job(self, job_id: str) -> JobRecord | None:
        rec = await self.get_job_any(job_id)
        if rec is None or rec.status not in ("queued", "running") or rec.attempts >= MAX_ATTEMPTS:
            return None
        rec.status, rec.attempts = "running", rec.attempts + 1
        return rec

    async def set_progress(self, job_id: str, progress: float) -> None:
        rec = await self.get_job_any(job_id)
        if rec:
            rec.progress = progress

    async def finish_job(
        self,
        job_id: str,
        status: str,
        error: str | None,
        creation_id: str | None,
        gpu_seconds: float,
    ) -> None:
        rec = await self.get_job_any(job_id)
        if rec:
            rec.status, rec.error, rec.creation_id = status, error, creation_id
            rec.progress = 1.0 if status == "succeeded" else rec.progress

    async def add_creation(self, tenant_id: str, owner_id: str, rec: CreationRecord) -> None:
        self.creations.append((tenant_id, owner_id, rec))

    async def list_creations(
        self, tenant_id: str, owner_id: str, limit: int
    ) -> list[CreationRecord]:
        mine = [c for t, o, c in self.creations if (t, o) == (tenant_id, owner_id)]
        return sorted(mine, key=lambda c: c.created_at, reverse=True)[:limit]

    async def get_creation(
        self, tenant_id: str, owner_id: str, creation_id: str
    ) -> CreationRecord | None:
        return next(
            (c for t, o, c in self.creations if (t, o, c.id) == (tenant_id, owner_id, creation_id)),
            None,
        )

    async def delete_creation(
        self, tenant_id: str, owner_id: str, creation_id: str
    ) -> CreationRecord | None:
        found = await self.get_creation(tenant_id, owner_id, creation_id)
        if found:
            self.creations = [x for x in self.creations if x[2].id != creation_id]
        return found


def _uuid(value: str) -> UUID | None:
    try:
        return UUID(value)
    except ValueError:
        return None


_JOB_COLS = (
    "id::text, tenant_id, owner_id, status, request, progress, error, creation_id::text, "
    "attempts, created_at"
)


def _job(r: Any) -> JobRecord:
    return JobRecord(
        id=r[0], tenant_id=r[1], owner_id=r[2], status=r[3], request=r[4], progress=float(r[5]),
        error=r[6], creation_id=r[7], attempts=r[8], created_at=r[9].isoformat(),
    )  # fmt: skip


class PostgresJobStore:
    def __init__(self, engine: AsyncEngine):
        self._engine = engine

    async def create_job(self, tenant_id: str, owner_id: str, request: dict[str, Any]) -> JobRecord:
        async with self._engine.begin() as conn:
            row = (
                await conn.execute(
                    text(
                        f"""
                        INSERT INTO jobs (tenant_id, product_id, owner_id, request)
                        VALUES (:t, :p, :o, CAST(:r AS jsonb)) RETURNING {_JOB_COLS}
                        """
                    ),
                    {"t": tenant_id, "p": PRODUCT_ID, "o": owner_id, "r": json.dumps(request)},
                )
            ).one()
        return _job(row)

    async def get_job(self, tenant_id: str, owner_id: str, job_id: str) -> JobRecord | None:
        if _uuid(job_id) is None:
            return None
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        f"""SELECT {_JOB_COLS} FROM jobs WHERE id = CAST(:i AS uuid)
                        AND tenant_id = :t AND product_id = :p AND owner_id = :o"""
                    ),
                    {"i": job_id, "t": tenant_id, "p": PRODUCT_ID, "o": owner_id},
                )
            ).first()
        return _job(row) if row else None

    async def get_job_any(self, job_id: str) -> JobRecord | None:
        if _uuid(job_id) is None:
            return None
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    text(f"SELECT {_JOB_COLS} FROM jobs WHERE id = CAST(:i AS uuid)"), {"i": job_id}
                )
            ).first()
        return _job(row) if row else None

    async def list_jobs(self, tenant_id: str, owner_id: str, limit: int) -> list[JobRecord]:
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        f"""SELECT {_JOB_COLS} FROM jobs WHERE tenant_id = :t AND product_id = :p
                        AND owner_id = :o ORDER BY created_at DESC LIMIT :n"""
                    ),
                    {"t": tenant_id, "p": PRODUCT_ID, "o": owner_id, "n": limit},
                )
            ).all()
        return [_job(r) for r in rows]

    async def cancel_queued(self, tenant_id: str, owner_id: str, job_id: str) -> JobRecord | None:
        if _uuid(job_id) is None:
            return None
        async with self._engine.begin() as conn:
            row = (
                await conn.execute(
                    text(
                        f"""
                        UPDATE jobs SET status = 'canceled', finished_at = now()
                        WHERE id = CAST(:i AS uuid) AND status = 'queued' AND tenant_id = :t
                          AND product_id = :p AND owner_id = :o
                        RETURNING {_JOB_COLS}
                        """
                    ),
                    {"i": job_id, "t": tenant_id, "p": PRODUCT_ID, "o": owner_id},
                )
            ).first()
        return _job(row) if row else None

    async def start_job(self, job_id: str) -> JobRecord | None:
        async with self._engine.begin() as conn:
            row = (
                await conn.execute(
                    text(
                        f"""
                        UPDATE jobs SET status = 'running', attempts = attempts + 1,
                               started_at = COALESCE(started_at, now())
                        WHERE id = CAST(:i AS uuid) AND status IN ('queued', 'running')
                          AND attempts < :max
                        RETURNING {_JOB_COLS}
                        """
                    ),
                    {"i": job_id, "max": MAX_ATTEMPTS},
                )
            ).first()
        return _job(row) if row else None

    async def set_progress(self, job_id: str, progress: float) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text("UPDATE jobs SET progress = :p WHERE id = CAST(:i AS uuid)"),
                {"i": job_id, "p": progress},
            )

    async def finish_job(
        self,
        job_id: str,
        status: str,
        error: str | None,
        creation_id: str | None,
        gpu_seconds: float,
    ) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    UPDATE jobs SET status = :s, error = :e, creation_id = CAST(:c AS uuid),
                           gpu_seconds = :g, finished_at = now(),
                           progress = CASE WHEN :s = 'succeeded' THEN 1 ELSE progress END
                    WHERE id = CAST(:i AS uuid)
                    """
                ),
                {"i": job_id, "s": status, "e": error, "c": creation_id, "g": gpu_seconds},
            )

    async def add_creation(self, tenant_id: str, owner_id: str, rec: CreationRecord) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO creations (id, tenant_id, product_id, owner_id, prompt, params,
                        provider_id, parent_id, image_key, thumb_key, width, height)
                    VALUES (CAST(:id AS uuid), :t, :p, :o, :prompt, CAST(:params AS jsonb), :prov,
                        CAST(:parent AS uuid), :ik, :tk, :w, :h)
                    """
                ),
                {
                    "id": rec.id, "t": tenant_id, "p": PRODUCT_ID, "o": owner_id,
                    "prompt": rec.prompt, "params": json.dumps(rec.params),
                    "prov": rec.provider_id, "parent": rec.parent_id, "ik": rec.image_key,
                    "tk": rec.thumb_key, "w": rec.width, "h": rec.height,
                },
            )  # fmt: skip
            for m in rec.models:
                await conn.execute(
                    text(
                        """
                        INSERT INTO creation_models
                            (creation_id, version_id, role, weight, name, family, rel_path,
                             header_sha256)
                        VALUES (CAST(:c AS uuid), CAST(:v AS uuid), :role, :w, :n, :f, :r, :s)
                        """
                    ),
                    {
                        "c": rec.id, "v": m.version_id, "role": m.role, "w": m.weight,
                        "n": m.name, "f": m.family, "r": m.rel_path, "s": m.header_sha256,
                    },
                )  # fmt: skip

    async def _creations(
        self, tenant_id: str, owner_id: str, where: str, params: dict[str, Any]
    ) -> list[CreationRecord]:
        base = {"t": tenant_id, "p": PRODUCT_ID, "o": owner_id, **params}
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        f"""
                        SELECT id::text, prompt, params, provider_id, image_key, thumb_key, width,
                               height, created_at, parent_id::text
                        FROM creations
                        WHERE tenant_id = :t AND product_id = :p AND owner_id = :o {where}
                        """
                    ),
                    base,
                )
            ).all()
            ids = [r[0] for r in rows]
            refs: dict[str, list[CreationModelRef]] = {}
            if ids:
                for m in (
                    await conn.execute(
                        text(
                            """
                            SELECT creation_id::text, version_id::text, name, family, rel_path,
                                   header_sha256, role, weight
                            FROM creation_models WHERE creation_id = ANY(CAST(:ids AS uuid[]))
                            ORDER BY (role = 'base') DESC, name
                            """
                        ),
                        {"ids": ids},
                    )
                ).all():
                    refs.setdefault(m[0], []).append(
                        CreationModelRef(
                            version_id=m[1],
                            name=m[2],
                            family=m[3],
                            rel_path=m[4],
                            header_sha256=m[5],
                            role=m[6],
                            weight=m[7],
                        )
                    )
        return [
            CreationRecord(
                id=r[0],
                prompt=r[1],
                params=r[2],
                provider_id=r[3],
                image_key=r[4],
                thumb_key=r[5],
                width=r[6],
                height=r[7],
                parent_id=r[9],
                created_at=r[8].isoformat(),
                models=refs.get(r[0], []),
            )  # fmt: skip
            for r in rows
        ]

    async def list_creations(
        self, tenant_id: str, owner_id: str, limit: int
    ) -> list[CreationRecord]:
        recs = await self._creations(tenant_id, owner_id, "", {})
        return sorted(recs, key=lambda c: c.created_at, reverse=True)[:limit]

    async def get_creation(
        self, tenant_id: str, owner_id: str, creation_id: str
    ) -> CreationRecord | None:
        if _uuid(creation_id) is None:
            return None
        found = await self._creations(
            tenant_id, owner_id, "AND id = CAST(:i AS uuid)", {"i": creation_id}
        )
        return found[0] if found else None

    async def delete_creation(
        self, tenant_id: str, owner_id: str, creation_id: str
    ) -> CreationRecord | None:
        found = await self.get_creation(tenant_id, owner_id, creation_id)
        if found is None:
            return None
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """DELETE FROM creations WHERE id = CAST(:i AS uuid) AND tenant_id = :t
                    AND product_id = :p AND owner_id = :o"""
                ),
                {"i": creation_id, "t": tenant_id, "p": PRODUCT_ID, "o": owner_id},
            )
        return found
