import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from wasifyai_art_contracts import UploadSummary

from wasifyai_art_api.identity import Identity, resolve_identity
from wasifyai_art_api.limits import RateLimiter
from wasifyai_art_api.routes import get_limiter, get_uploads
from wasifyai_art_api.routes_generation import get_storage
from wasifyai_art_api.settings import Settings, get_app_settings
from wasifyai_art_api.storage import Storage
from wasifyai_art_api.uploads import UploadRefused, UploadStore, new_upload, release, sanitize

router = APIRouter()


@router.post("/v1/uploads/images", response_model=UploadSummary, status_code=201, tags=["uploads"])
async def upload_image(
    request: Request,
    identity: Identity = Depends(resolve_identity),
    uploads: UploadStore = Depends(get_uploads),
    storage: Storage = Depends(get_storage),
    limiter: RateLimiter = Depends(get_limiter),
    settings: Settings = Depends(get_app_settings),
) -> UploadSummary:
    """The raw picture is the request body.

    It is re-encoded, stripped of metadata and stored under the caller's own prefix; only an
    `upload_id` is returned."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > settings.upload_max_bytes:
        raise HTTPException(413, "That picture is too large (over 10 MB).")
    if not (request.headers.get("content-type") or "").startswith("image/"):
        raise HTTPException(415, "Send the picture as an image/* request body.")
    if not await limiter.hit(f"upload:{identity.tenant_id}:{identity.user_id}", 60, 3600):
        raise HTTPException(429, "Too many uploads. Try again later.")
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():  # a hard cap while reading, not trusting Content-Length
        size += len(chunk)
        if size > settings.upload_max_bytes:
            raise HTTPException(413, "That picture is too large (over 10 MB).")
        chunks.append(chunk)
    try:
        png, width, height = await asyncio.to_thread(
            sanitize, b"".join(chunks), settings.upload_max_side
        )
    except UploadRefused as e:
        raise HTTPException(e.status, e.message) from None
    rec = new_upload(identity.user_id, identity.tenant_id, png, width, height)
    await storage.put(identity.tenant_id, identity.user_id, rec.key, png)
    await uploads.add(identity.tenant_id, identity.user_id, rec)
    return UploadSummary(upload_id=rec.id, width=width, height=height, bytes=len(png))


@router.delete("/v1/uploads/{upload_id}", status_code=204, tags=["uploads"])
async def delete_upload(
    upload_id: str,
    identity: Identity = Depends(resolve_identity),
    uploads: UploadStore = Depends(get_uploads),
    storage: Storage = Depends(get_storage),
) -> Response:
    rec = await uploads.get(identity.tenant_id, identity.user_id, upload_id)
    if rec is None:
        raise HTTPException(404, "upload not found")
    await release(uploads, storage, rec)
    return Response(status_code=204)
