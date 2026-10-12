"""Hub storage. Every query is scoped by tenant, product and owner (rule 4 of CLAUDE.md)."""

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from wasifyai_art_contracts import (
    ModelFileSummary,
    ModelSummary,
    ModelVersionSummary,
    VersionUpdate,
)

from wasifyai_art_api import PRODUCT_ID
from wasifyai_art_api.plan import Plan, StoredFile
from wasifyai_art_api.scanner import ScannedFile


class VersionConflict(ValueError):
    """The requested change is not allowed (for example `ready` without a family)."""


@dataclass(frozen=True)
class Hint:
    name: str
    family_hint: str | None
    trigger_hints: list[str]


@dataclass(frozen=True)
class VersionDetail:
    version_id: str
    name: str
    kind: str
    family: str | None
    status: str
    rel_path: str | None  # the version's first present file
    header_sha256: str | None


class ModelStore(Protocol):
    async def get_version_detail(
        self, tenant_id: str, owner_id: str, version_id: str
    ) -> VersionDetail | None: ...

    async def list_models(self, tenant_id: str, owner_id: str) -> list[ModelSummary]: ...

    async def list_files(self, tenant_id: str, owner_id: str) -> list[StoredFile]: ...

    async def apply_plan(
        self, tenant_id: str, owner_id: str, plan: Plan, hints: dict[str, Hint]
    ) -> None: ...

    async def update_version(
        self, tenant_id: str, owner_id: str, version_id: str, upd: VersionUpdate
    ) -> ModelVersionSummary | None: ...


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60] or "model"


def decide_update(current: dict[str, Any], upd: VersionUpdate) -> dict[str, Any]:
    """The version's new family, trigger words and status, or VersionConflict. Pure."""
    family = upd.family if upd.family is not None else current["family"]
    words = upd.trigger_words if upd.trigger_words is not None else current["trigger_words"]
    words = [w.strip() for w in words if w.strip()]
    status = upd.status if upd.status is not None else current["status"]
    if status == "ready":
        if not family:
            raise VersionConflict("choose a base family before marking a model ready")
        if current["files_missing"]:
            raise VersionConflict("the file is missing from disk")
    elif status == "missing" and upd.status is not None:
        raise VersionConflict("status 'missing' is set by scanning only")
    return {"family": family or None, "trigger_words": words, "status": status}


def _version_status_after_missing(status: str) -> str:
    return status if status == "disabled" else "missing"


def _version_status_after_restore(status: str, family: str | None, same: bool) -> str:
    if status != "missing":
        return status
    return "ready" if family and same else "unconfirmed"


class InMemoryModelStore:
    """For tests. Mirrors the Postgres behaviour, including the scoping."""

    def __init__(self) -> None:
        self._models: list[dict[str, Any]] = []
        self._versions: list[dict[str, Any]] = []
        self._files: list[dict[str, Any]] = []

    def add_summary(self, tenant_id: str, owner_id: str, model: ModelSummary) -> None:
        self._models.append({"t": tenant_id, "o": owner_id, "m": model})

    async def get_version_detail(
        self, tenant_id: str, owner_id: str, version_id: str
    ) -> VersionDetail | None:
        v = next((x for x in self._versions if x["id"] == version_id), None)
        if v is None:
            return None
        m = next(
            (
                r["m"]
                for r in self._models
                if r["m"].id == v["model_id"] and (r["t"], r["o"]) == (tenant_id, owner_id)
            ),
            None,
        )
        if m is None:
            return None
        f = next(
            (x for x in self._files if x["version_id"] == version_id and x["status"] == "present"),
            None,
        )
        return VersionDetail(
            version_id, m.name, m.kind, v["family"], v["status"],
            f["rel_path"] if f else None, f["header"] if f else None,
        )  # fmt: skip

    async def list_models(self, tenant_id: str, owner_id: str) -> list[ModelSummary]:
        out: list[ModelSummary] = []
        for row in self._models:
            if (row["t"], row["o"]) != (tenant_id, owner_id):
                continue
            m: ModelSummary = row["m"]
            versions = list(m.versions)
            for v in self._versions:
                if v["model_id"] == m.id:
                    files = [
                        ModelFileSummary(
                            rel_path=f["rel_path"],
                            folder=f["folder"],
                            size_bytes=f["size_bytes"],
                            status=f["status"],
                        )  # fmt: skip
                        for f in self._files
                        if f["version_id"] == v["id"]
                    ]
                    versions.append(
                        ModelVersionSummary(
                            id=v["id"],
                            label=v["label"],
                            family=v["family"],
                            status=v["status"],
                            trigger_words=list(v["trigger_words"]),
                            family_hint=v["family_hint"],
                            trigger_hints=list(v["trigger_hints"]),
                            files=files,
                        )  # fmt: skip
                    )
            out.append(m.model_copy(update={"versions": versions}))
        return out

    async def list_files(self, tenant_id: str, owner_id: str) -> list[StoredFile]:
        return [
            StoredFile(
                f["id"], f["version_id"], f["rel_path"], f["status"], f["header"], f["size_bytes"]
            )
            for f in self._files
            if (f["t"], f["o"]) == (tenant_id, owner_id)
        ]

    def _version(self, version_id: str) -> dict[str, Any]:
        return next(v for v in self._versions if v["id"] == version_id)

    async def apply_plan(
        self, tenant_id: str, owner_id: str, plan: Plan, hints: dict[str, Hint]
    ) -> None:
        def file_fields(s: ScannedFile) -> dict[str, Any]:
            return {"size_bytes": s.size_bytes, "mtime": s.mtime, "header": s.header.header_sha256}

        for s in plan.add:
            h = hints[s.rel_path]
            taken = {m["m"].slug for m in self._models if m["t"] == tenant_id}
            slug, n = slugify(h.name), 2
            base = slug
            while slug in taken:
                slug, n = f"{base}-{n}", n + 1
            model_id, version_id = str(uuid4()), str(uuid4())
            self._models.append(
                {
                    "t": tenant_id,
                    "o": owner_id,
                    "m": ModelSummary(id=model_id, kind=s.kind, name=h.name, slug=slug),
                }  # fmt: skip
            )
            self._versions.append(
                {
                    "id": version_id,
                    "model_id": model_id,
                    "label": "v1",
                    "family": None,
                    "status": "unconfirmed",
                    "trigger_words": [],
                    "family_hint": h.family_hint,
                    "trigger_hints": h.trigger_hints,
                }  # fmt: skip
            )
            self._files.append(
                {
                    "id": str(uuid4()),
                    "t": tenant_id,
                    "o": owner_id,
                    "version_id": version_id,
                    "rel_path": s.rel_path,
                    "folder": s.folder,
                    "status": "present",
                    **file_fields(s),
                }  # fmt: skip
            )
        for stored, s in plan.update:
            f = next(x for x in self._files if x["id"] == stored.file_id)
            f.update(file_fields(s))
            v = self._version(stored.version_id)
            h = hints[s.rel_path]
            v.update(family_hint=h.family_hint, trigger_hints=h.trigger_hints)
            if v["status"] != "disabled":
                v["status"] = "unconfirmed"
        for stored in plan.missing:
            f = next(x for x in self._files if x["id"] == stored.file_id)
            f["status"] = "missing"
            v = self._version(stored.version_id)
            v["status"] = _version_status_after_missing(v["status"])
        for stored, s, same in plan.restore:
            f = next(x for x in self._files if x["id"] == stored.file_id)
            f.update(file_fields(s), status="present")
            v = self._version(stored.version_id)
            v["status"] = _version_status_after_restore(v["status"], v["family"], same)

    async def update_version(
        self, tenant_id: str, owner_id: str, version_id: str, upd: VersionUpdate
    ) -> ModelVersionSummary | None:
        owned = {m["m"].id for m in self._models if (m["t"], m["o"]) == (tenant_id, owner_id)}
        v = next((x for x in self._versions if x["id"] == version_id), None)
        if v is None or v["model_id"] not in owned:
            return None
        missing = any(
            f["status"] == "missing" for f in self._files if f["version_id"] == version_id
        )
        new = decide_update({**v, "files_missing": missing}, upd)
        v.update(new)
        for m in await self.list_models(tenant_id, owner_id):
            for ver in m.versions:
                if ver.id == version_id:
                    return ver
        return None


class PostgresModelStore:
    def __init__(self, engine: AsyncEngine):
        self._engine = engine

    async def get_version_detail(
        self, tenant_id: str, owner_id: str, version_id: str
    ) -> VersionDetail | None:
        try:
            UUID(version_id)
        except ValueError:
            return None
        async with self._engine.connect() as conn:
            r = (
                await conn.execute(
                    text(
                        """
                        SELECT v.id::text, m.name, m.kind, v.family, v.status,
                               (SELECT f.rel_path FROM model_files f
                                WHERE f.version_id = v.id AND f.status = 'present'
                                ORDER BY f.rel_path LIMIT 1),
                               (SELECT f.header_sha256 FROM model_files f
                                WHERE f.version_id = v.id AND f.status = 'present'
                                ORDER BY f.rel_path LIMIT 1)
                        FROM model_versions v JOIN models m ON m.id = v.model_id
                        WHERE v.id = CAST(:v AS uuid) AND m.tenant_id = :t AND m.product_id = :p
                          AND m.owner_id = :o
                        """
                    ),
                    {"v": version_id, "t": tenant_id, "p": PRODUCT_ID, "o": owner_id},
                )
            ).first()
        return VersionDetail(*r) if r else None

    async def list_models(self, tenant_id: str, owner_id: str) -> list[ModelSummary]:
        p = {"t": tenant_id, "p": PRODUCT_ID, "o": owner_id}
        async with self._engine.connect() as conn:
            models = (
                await conn.execute(
                    text(
                        """
                        SELECT id::text, kind, name, slug, tags, visibility FROM models
                        WHERE tenant_id = :t AND product_id = :p AND owner_id = :o
                        ORDER BY created_at DESC, id DESC
                        """
                    ),
                    p,
                )
            ).all()
            versions = (
                await conn.execute(
                    text(
                        """
                        SELECT v.id::text, v.model_id::text, v.label, v.family, v.status,
                               v.trigger_words, v.family_hint, v.trigger_hints
                        FROM model_versions v JOIN models m ON m.id = v.model_id
                        WHERE m.tenant_id = :t AND m.product_id = :p AND m.owner_id = :o
                        ORDER BY v.created_at DESC, v.id DESC
                        """
                    ),
                    p,
                )
            ).all()
            files = (
                await conn.execute(
                    text(
                        """
                        SELECT version_id::text, rel_path, folder, size_bytes, status
                        FROM model_files
                        WHERE tenant_id = :t AND product_id = :p AND owner_id = :o
                        ORDER BY rel_path
                        """
                    ),
                    p,
                )
            ).all()
        files_by_version: dict[str, list[ModelFileSummary]] = {}
        for f in files:
            files_by_version.setdefault(f[0], []).append(
                ModelFileSummary(rel_path=f[1], folder=f[2], size_bytes=f[3], status=f[4])
            )
        by_model: dict[str, list[ModelVersionSummary]] = {}
        for v in versions:
            by_model.setdefault(v[1], []).append(
                ModelVersionSummary(
                    id=v[0],
                    label=v[2],
                    family=v[3],
                    status=v[4],
                    trigger_words=list(v[5] or []),
                    family_hint=v[6],
                    trigger_hints=list(v[7] or []),
                    files=files_by_version.get(v[0], []),
                )  # fmt: skip
            )
        return [
            ModelSummary(
                id=m[0],
                kind=m[1],
                name=m[2],
                slug=m[3],
                tags=list(m[4] or []),
                visibility=m[5],
                versions=by_model.get(m[0], []),
            )  # fmt: skip
            for m in models
        ]

    async def list_files(self, tenant_id: str, owner_id: str) -> list[StoredFile]:
        async with self._engine.connect() as conn:
            rows = await conn.execute(
                text(
                    """
                    SELECT id::text, version_id::text, rel_path, status, header_sha256, size_bytes
                    FROM model_files
                    WHERE tenant_id = :t AND product_id = :p AND owner_id = :o
                    """
                ),
                {"t": tenant_id, "p": PRODUCT_ID, "o": owner_id},
            )
            return [StoredFile(r[0], r[1], r[2], r[3], r[4] or "", int(r[5] or 0)) for r in rows]

    async def apply_plan(
        self, tenant_id: str, owner_id: str, plan: Plan, hints: dict[str, Hint]
    ) -> None:
        base = {"t": tenant_id, "p": PRODUCT_ID, "o": owner_id}

        def file_params(s: ScannedFile) -> dict[str, Any]:
            return {
                "size": s.size_bytes, "mtime": s.mtime, "sha": s.header.header_sha256,
                "n": s.header.tensor_count, "meta": json.dumps(s.header.metadata),
            }  # fmt: skip

        async with self._engine.begin() as conn:
            for s in plan.add:
                h = hints[s.rel_path]
                slug = await self._unique_slug(conn, tenant_id, slugify(h.name))
                model_id = (
                    await conn.execute(
                        text(
                            """
                            INSERT INTO models (tenant_id, product_id, owner_id, kind, name, slug)
                            VALUES (:t, :p, :o, :kind, :name, :slug) RETURNING id
                            """
                        ),
                        {**base, "kind": s.kind, "name": h.name, "slug": slug},
                    )
                ).scalar_one()
                version_id = (
                    await conn.execute(
                        text(
                            """
                            INSERT INTO model_versions
                              (model_id, tenant_id, product_id, label, family_hint, trigger_hints,
                               status, source)
                            VALUES (:m, :t, :p, 'v1', :fh, :th, 'unconfirmed', 'scan') RETURNING id
                            """
                        ),
                        {
                            "m": model_id,
                            "t": tenant_id,
                            "p": PRODUCT_ID,
                            "fh": h.family_hint,
                            "th": h.trigger_hints,
                        },
                    )  # fmt: skip
                ).scalar_one()
                await conn.execute(
                    text(
                        """
                        INSERT INTO model_files
                          (version_id, tenant_id, product_id, owner_id, rel_path, folder,
                           size_bytes, mtime, header_sha256, tensor_count, meta)
                        VALUES (:v, :t, :p, :o, :rel, :folder, :size, :mtime, :sha, :n,
                                CAST(:meta AS jsonb))
                        """
                    ),
                    {
                        **base,
                        "v": version_id,
                        "rel": s.rel_path,
                        "folder": s.folder,
                        **file_params(s),
                    },
                )
            for stored, s in plan.update:
                h = hints[s.rel_path]
                await self._set_file(conn, stored.file_id, s, "present")
                await conn.execute(
                    text(
                        """
                        UPDATE model_versions SET family_hint = :fh, trigger_hints = :th,
                               status = CASE WHEN status = 'disabled' THEN status
                                             ELSE 'unconfirmed' END
                        WHERE id = CAST(:v AS uuid)
                        """
                    ),
                    {"v": stored.version_id, "fh": h.family_hint, "th": h.trigger_hints},
                )
            for stored in plan.missing:
                await conn.execute(
                    text("UPDATE model_files SET status = 'missing' WHERE id = CAST(:f AS uuid)"),
                    {"f": stored.file_id},
                )
                await conn.execute(
                    text(
                        """
                        UPDATE model_versions
                        SET status = CASE WHEN status = 'disabled' THEN status ELSE 'missing' END
                        WHERE id = CAST(:v AS uuid)
                        """
                    ),
                    {"v": stored.version_id},
                )
            for stored, s, same in plan.restore:
                await self._set_file(conn, stored.file_id, s, "present")
                await conn.execute(
                    text(
                        """
                        UPDATE model_versions SET status = CASE
                          WHEN status <> 'missing' THEN status
                          WHEN family IS NOT NULL AND :same THEN 'ready'
                          ELSE 'unconfirmed' END
                        WHERE id = CAST(:v AS uuid)
                        """
                    ),
                    {"v": stored.version_id, "same": same},
                )

    @staticmethod
    async def _unique_slug(conn: Any, tenant_id: str, base: str) -> str:
        rows = await conn.execute(
            text(
                "SELECT slug FROM models "
                "WHERE tenant_id = :t AND product_id = :p AND slug LIKE :like"
            ),
            {"t": tenant_id, "p": PRODUCT_ID, "like": base + "%"},
        )
        taken = {r[0] for r in rows}
        slug, n = base, 2
        while slug in taken:
            slug, n = f"{base}-{n}", n + 1
        return slug

    @staticmethod
    async def _set_file(conn: Any, file_id: str, s: ScannedFile, status: str) -> None:
        await conn.execute(
            text(
                """
                UPDATE model_files SET size_bytes = :size, mtime = :mtime, header_sha256 = :sha,
                       tensor_count = :n, meta = CAST(:meta AS jsonb), status = :status
                WHERE id = CAST(:f AS uuid)
                """
            ),
            {
                "f": file_id, "size": s.size_bytes, "mtime": s.mtime, "sha": s.header.header_sha256,
                "n": s.header.tensor_count, "meta": json.dumps(s.header.metadata), "status": status,
            },
        )  # fmt: skip

    async def update_version(
        self, tenant_id: str, owner_id: str, version_id: str, upd: VersionUpdate
    ) -> ModelVersionSummary | None:
        try:
            UUID(version_id)
        except ValueError:
            return None
        async with self._engine.begin() as conn:
            row = (
                await conn.execute(
                    text(
                        """
                        SELECT v.family, v.trigger_words, v.status,
                               EXISTS (SELECT 1 FROM model_files f
                                       WHERE f.version_id = v.id AND f.status = 'missing')
                        FROM model_versions v JOIN models m ON m.id = v.model_id
                        WHERE v.id = CAST(:v AS uuid) AND m.tenant_id = :t AND m.product_id = :p
                          AND m.owner_id = :o
                        FOR UPDATE OF v
                        """
                    ),
                    {"v": version_id, "t": tenant_id, "p": PRODUCT_ID, "o": owner_id},
                )
            ).first()
            if row is None:
                return None
            new = decide_update(
                {"family": row[0], "trigger_words": list(row[1] or []), "status": row[2],
                 "files_missing": bool(row[3])},
                upd,
            )  # fmt: skip
            await conn.execute(
                text(
                    """
                    UPDATE model_versions SET family = :f, trigger_words = :w, status = :s
                    WHERE id = CAST(:v AS uuid)
                    """
                ),
                {
                    "v": version_id,
                    "f": new["family"],
                    "w": new["trigger_words"],
                    "s": new["status"],
                },
            )
        for m in await self.list_models(tenant_id, owner_id):
            for ver in m.versions:
                if ver.id == version_id:
                    return ver
        return None
