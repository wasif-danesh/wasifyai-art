"""Pictures a user uploads as the input of an edit.

Uploaded files are untrusted and often personal photos, so what is stored is not what was sent:
the picture is decoded, rotated by its EXIF orientation, stripped of all metadata (GPS, camera,
software), scaled so its longest side is at most `max_side` and saved as a PNG. The client's file
name is never used."""

import io
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import UUID, uuid4

from PIL import Image, ImageOps, UnidentifiedImageError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from wasifyai_art_api import PRODUCT_ID
from wasifyai_art_api.storage import Storage

ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}
MAX_PIXELS = 24_000_000


class UploadRefused(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def sanitize(data: bytes, max_side: int = 2048) -> tuple[bytes, int, int]:
    """The stored PNG and its size, or UploadRefused.

    The format is decided by decoding, never by the declared type or file name; SVG, GIF and
    anything with scripts are refused."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ALLOWED_FORMATS:
                raise UploadRefused(415, "Only PNG, JPEG and WebP pictures are accepted.")
            if im.width * im.height > MAX_PIXELS:
                raise UploadRefused(422, "That picture is too large (over 24 megapixels).")
            im.load()  # fully decode: a truncated or corrupt file fails here
            upright = ImageOps.exif_transpose(im)
            rgb = upright.convert("RGB")  # also drops alpha and every metadata chunk
    except UploadRefused:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise UploadRefused(422, "That file could not be read as a picture.") from None
    rgb.thumbnail((max_side, max_side))
    out = io.BytesIO()
    rgb.save(out, "PNG", optimize=True)  # no exif, no text chunks
    return out.getvalue(), rgb.width, rgb.height


@dataclass
class UploadRecord:
    id: str
    tenant_id: str
    owner_id: str
    key: str  # storage path relative to the owner's prefix
    bytes: int
    width: int
    height: int
    created_at: datetime
    consumed: bool = False


class UploadStore(Protocol):
    async def add(self, tenant_id: str, owner_id: str, rec: UploadRecord) -> None: ...

    async def get(self, tenant_id: str, owner_id: str, upload_id: str) -> UploadRecord | None: ...

    async def consume(self, tenant_id: str, owner_id: str, upload_id: str) -> None: ...

    async def delete(self, tenant_id: str, owner_id: str, upload_id: str) -> None: ...

    async def list_stale(self, before: datetime) -> list[UploadRecord]: ...


class InMemoryUploadStore:
    def __init__(self) -> None:
        self.rows: dict[str, UploadRecord] = {}

    async def add(self, tenant_id: str, owner_id: str, rec: UploadRecord) -> None:
        self.rows[rec.id] = rec

    async def get(self, tenant_id: str, owner_id: str, upload_id: str) -> UploadRecord | None:
        r = self.rows.get(upload_id)
        return r if r and (r.tenant_id, r.owner_id) == (tenant_id, owner_id) else None

    async def consume(self, tenant_id: str, owner_id: str, upload_id: str) -> None:
        if r := await self.get(tenant_id, owner_id, upload_id):
            r.consumed = True

    async def delete(self, tenant_id: str, owner_id: str, upload_id: str) -> None:
        if await self.get(tenant_id, owner_id, upload_id):
            del self.rows[upload_id]

    async def list_stale(self, before: datetime) -> list[UploadRecord]:
        return [r for r in self.rows.values() if not r.consumed and r.created_at < before]


class PostgresUploadStore:
    def __init__(self, engine: AsyncEngine):
        self._engine = engine

    async def add(self, tenant_id: str, owner_id: str, rec: UploadRecord) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO uploads
                        (id, tenant_id, product_id, owner_id, key, bytes, width, height)
                    VALUES (CAST(:i AS uuid), :t, :p, :o, :k, :b, :w, :h)
                    """
                ),
                {"i": rec.id, "t": tenant_id, "p": PRODUCT_ID, "o": owner_id, "k": rec.key,
                 "b": rec.bytes, "w": rec.width, "h": rec.height},
            )  # fmt: skip

    @staticmethod
    def _rec(r: Any) -> UploadRecord:
        return UploadRecord(
            id=r[0], tenant_id=r[1], owner_id=r[2], key=r[3], bytes=r[4], width=r[5], height=r[6],
            created_at=r[7], consumed=r[8] is not None,
        )  # fmt: skip

    _COLS = "id::text, tenant_id, owner_id, key, bytes, width, height, created_at, consumed_at"

    async def get(self, tenant_id: str, owner_id: str, upload_id: str) -> UploadRecord | None:
        try:
            UUID(upload_id)
        except ValueError:
            return None
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        f"""SELECT {self._COLS} FROM uploads WHERE id = CAST(:i AS uuid)
                        AND tenant_id = :t AND product_id = :p AND owner_id = :o"""
                    ),
                    {"i": upload_id, "t": tenant_id, "p": PRODUCT_ID, "o": owner_id},
                )
            ).first()
        return self._rec(row) if row else None

    async def consume(self, tenant_id: str, owner_id: str, upload_id: str) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """UPDATE uploads SET consumed_at = now() WHERE id = CAST(:i AS uuid)
                    AND tenant_id = :t AND product_id = :p AND owner_id = :o"""
                ),
                {"i": upload_id, "t": tenant_id, "p": PRODUCT_ID, "o": owner_id},
            )

    async def delete(self, tenant_id: str, owner_id: str, upload_id: str) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """DELETE FROM uploads WHERE id = CAST(:i AS uuid) AND tenant_id = :t
                    AND product_id = :p AND owner_id = :o"""
                ),
                {"i": upload_id, "t": tenant_id, "p": PRODUCT_ID, "o": owner_id},
            )

    async def list_stale(self, before: datetime) -> list[UploadRecord]:
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        f"""SELECT {self._COLS} FROM uploads
                        WHERE consumed_at IS NULL AND created_at < :b AND product_id = :p"""
                    ),
                    {"b": before, "p": PRODUCT_ID},
                )
            ).all()
        return [self._rec(r) for r in rows]


def new_upload(owner: str, tenant: str, png: bytes, width: int, height: int) -> UploadRecord:
    uid = str(uuid4())
    return UploadRecord(
        id=uid, tenant_id=tenant, owner_id=owner, key=f"uploads/{uid}.png", bytes=len(png),
        width=width, height=height, created_at=datetime.now(UTC),
    )  # fmt: skip


async def release(uploads: UploadStore, storage: Storage, rec: UploadRecord) -> None:
    """Remove the file and the record: an uploaded picture is not kept past its use."""
    await storage.delete(rec.tenant_id, rec.owner_id, rec.key)
    await uploads.delete(rec.tenant_id, rec.owner_id, rec.id)


async def sweep_uploads(
    uploads: UploadStore, storage: Storage, older_than: timedelta = timedelta(hours=24)
) -> int:
    """Delete pictures that were uploaded but never used (a closed tab). Returns how many."""
    stale = await uploads.list_stale(datetime.now(UTC) - older_than)
    for rec in stale:
        await release(uploads, storage, rec)
    return len(stale)
