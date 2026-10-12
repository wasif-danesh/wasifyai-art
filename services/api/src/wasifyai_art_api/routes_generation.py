import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from wasifyai_art_contracts import (
    CreationList,
    CreationSummary,
    GenerationRequest,
    JobList,
    JobSummary,
)

from wasifyai_art_api.bus import TERMINAL, EventBus
from wasifyai_art_api.families import FamilyConfig
from wasifyai_art_api.generation import GenerationRefused, resolve_request
from wasifyai_art_api.identity import Identity, resolve_identity
from wasifyai_art_api.jobs import CreationRecord, JobRecord, JobStore
from wasifyai_art_api.providers import ProviderConfig, describe
from wasifyai_art_api.routes import get_families, get_providers, get_store, get_uploads
from wasifyai_art_api.settings import Settings, get_app_settings
from wasifyai_art_api.storage import Storage
from wasifyai_art_api.store import ModelStore
from wasifyai_art_api.uploads import UploadStore
from wasifyai_art_api.workflows import WorkflowRegistry

router = APIRouter()


def get_jobs(request: Request) -> JobStore:
    return request.app.state.jobs


def get_bus(request: Request) -> EventBus:
    return request.app.state.bus


def get_storage(request: Request) -> Storage:
    return request.app.state.storage


def get_workflows(request: Request) -> WorkflowRegistry:
    return request.app.state.workflows


def job_summary(j: JobRecord) -> JobSummary:
    return JobSummary(
        id=j.id, status=j.status, progress=j.progress, error=j.error,  # type: ignore[arg-type]
        creation_id=j.creation_id, created_at=j.created_at, prompt=j.request.get("prompt", ""),
        mode=j.request.get("mode", "text"), model_name=j.request.get("model_name", ""),
    )  # fmt: skip


def creation_summary(c: CreationRecord) -> CreationSummary:
    return CreationSummary(
        id=c.id, prompt=c.prompt, width=c.width, height=c.height, provider_id=c.provider_id,
        created_at=c.created_at, parent_id=c.parent_id, image_url=f"/v1/creations/{c.id}/image",
        thumb_url=f"/v1/creations/{c.id}/thumb", params=c.params, models=c.models,
    )  # fmt: skip


@router.post("/v1/generations", response_model=JobSummary, status_code=202, tags=["generation"])
async def create_generation(
    body: GenerationRequest,
    identity: Identity = Depends(resolve_identity),
    store: ModelStore = Depends(get_store),
    jobs: JobStore = Depends(get_jobs),
    uploads: UploadStore = Depends(get_uploads),
    bus: EventBus = Depends(get_bus),
    configs: list[ProviderConfig] = Depends(get_providers),
    families: list[FamilyConfig] = Depends(get_families),
    workflows: WorkflowRegistry = Depends(get_workflows),
    settings: Settings = Depends(get_app_settings),
) -> JobSummary:
    """Validate and resolve the request, store a job and queue it. Never awaits GPU work."""
    address = settings.comfyui_base_url or ("stub" if settings.comfyui_mode == "stub" else "")
    providers = describe(configs, {"COMFYUI_BASE_URL": address})
    try:
        request = await resolve_request(
            body,
            identity.tenant_id,
            identity.user_id,
            store,
            jobs,
            uploads,
            providers,
            families,
            workflows,
            (settings.lora_weight_min, settings.lora_weight_max),
        )
    except GenerationRefused as e:
        raise HTTPException(e.status, e.message) from None
    job = await jobs.create_job(identity.tenant_id, identity.user_id, request)
    await bus.publish(job.id, "queued", {})
    await bus.enqueue(job.id)  # type: ignore[attr-defined]
    return job_summary(job)


@router.get("/v1/jobs", response_model=JobList, tags=["generation"])
async def list_jobs(
    limit: int = 30,
    identity: Identity = Depends(resolve_identity),
    jobs: JobStore = Depends(get_jobs),
) -> JobList:
    recs = await jobs.list_jobs(identity.tenant_id, identity.user_id, max(1, min(limit, 100)))
    return JobList(jobs=[job_summary(j) for j in recs])


@router.post("/v1/jobs/{job_id}/cancel", response_model=JobSummary, tags=["generation"])
async def cancel_job(
    job_id: str,
    identity: Identity = Depends(resolve_identity),
    jobs: JobStore = Depends(get_jobs),
    bus: EventBus = Depends(get_bus),
) -> JobSummary:
    """A queued job is canceled at once; a running one is asked to stop (it ends as `canceled`)."""
    job = await jobs.get_job(identity.tenant_id, identity.user_id, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.status in TERMINAL:
        raise HTTPException(409, "the job has already finished")
    await bus.request_cancel(job_id)
    queued = await jobs.cancel_queued(identity.tenant_id, identity.user_id, job_id)
    if queued is not None:
        await bus.publish(job_id, "canceled", {})
        return job_summary(queued)
    return job_summary(job)


@router.post(
    "/v1/jobs/{job_id}/retry", response_model=JobSummary, status_code=202, tags=["generation"]
)
async def retry_job(
    job_id: str,
    identity: Identity = Depends(resolve_identity),
    jobs: JobStore = Depends(get_jobs),
    bus: EventBus = Depends(get_bus),
) -> JobSummary:
    """Run a failed or canceled job again with the same resolved request, as a new job."""
    job = await jobs.get_job(identity.tenant_id, identity.user_id, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.status not in ("failed", "canceled"):
        raise HTTPException(409, "only a failed or canceled job can be retried")
    fresh = await jobs.create_job(identity.tenant_id, identity.user_id, dict(job.request))
    await bus.publish(fresh.id, "queued", {})
    await bus.enqueue(fresh.id)  # type: ignore[attr-defined]
    return job_summary(fresh)


@router.get("/v1/jobs/{job_id}", response_model=JobSummary, tags=["generation"])
async def get_job(
    job_id: str, identity: Identity = Depends(resolve_identity), jobs: JobStore = Depends(get_jobs)
) -> JobSummary:
    job = await jobs.get_job(identity.tenant_id, identity.user_id, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job_summary(job)


@router.get("/v1/jobs/{job_id}/events", tags=["generation"])
async def job_events(
    job_id: str,
    request: Request,
    identity: Identity = Depends(resolve_identity),
    jobs: JobStore = Depends(get_jobs),
    bus: EventBus = Depends(get_bus),
) -> StreamingResponse:
    """Server-sent events for one job. `Last-Event-ID` resumes after a dropped connection."""
    if await jobs.get_job(identity.tenant_id, identity.user_id, job_id) is None:
        raise HTTPException(404, "job not found")
    last = request.headers.get("last-event-id") or "0"

    async def stream() -> AsyncIterator[str]:
        cursor = last
        while True:
            events = await bus.read(job_id, cursor, 10_000)
            for e in events:
                cursor = e.id
                yield f"id: {e.id}\nevent: {e.type}\ndata: {json.dumps(e.data)}\n\n"
                if e.type in TERMINAL:
                    return
            if not events:
                job = await jobs.get_job(identity.tenant_id, identity.user_id, job_id)
                if job is None or job.status in TERMINAL:  # the event log may have expired
                    if job and job.status == "succeeded":
                        data = {"creation_id": job.creation_id}
                    else:
                        data = {"message": (job.error if job else None) or "The job is gone."}
                    yield f"event: {job.status if job else 'failed'}\ndata: {json.dumps(data)}\n\n"
                    return
                yield ": keepalive\n\n"

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


@router.get("/v1/creations", response_model=CreationList, tags=["creations"])
async def list_creations(
    limit: int = 48,
    identity: Identity = Depends(resolve_identity),
    jobs: JobStore = Depends(get_jobs),
) -> CreationList:
    recs = await jobs.list_creations(identity.tenant_id, identity.user_id, max(1, min(limit, 100)))
    return CreationList(creations=[creation_summary(c) for c in recs])


@router.get("/v1/creations/{creation_id}", response_model=CreationSummary, tags=["creations"])
async def get_creation(
    creation_id: str,
    identity: Identity = Depends(resolve_identity),
    jobs: JobStore = Depends(get_jobs),
) -> CreationSummary:
    rec = await jobs.get_creation(identity.tenant_id, identity.user_id, creation_id)
    if rec is None:
        raise HTTPException(404, "creation not found")
    return creation_summary(rec)


async def _file(
    creation_id: str, which: str, identity: Identity, jobs: JobStore, storage: Storage
) -> Response:
    rec = await jobs.get_creation(identity.tenant_id, identity.user_id, creation_id)
    if rec is None:
        raise HTTPException(404, "creation not found")
    key, media = (rec.image_key, "image/png") if which == "image" else (rec.thumb_key, "image/jpeg")
    try:
        data = await storage.get(identity.tenant_id, identity.user_id, key)
    except FileNotFoundError:
        raise HTTPException(404, "file not found") from None
    return Response(
        data,
        media_type=media,
        headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"},
    )


@router.get("/v1/creations/{creation_id}/image", tags=["creations"])
async def creation_image(
    creation_id: str,
    identity: Identity = Depends(resolve_identity),
    jobs: JobStore = Depends(get_jobs),
    storage: Storage = Depends(get_storage),
) -> Response:
    return await _file(creation_id, "image", identity, jobs, storage)


@router.get("/v1/creations/{creation_id}/thumb", tags=["creations"])
async def creation_thumb(
    creation_id: str,
    identity: Identity = Depends(resolve_identity),
    jobs: JobStore = Depends(get_jobs),
    storage: Storage = Depends(get_storage),
) -> Response:
    return await _file(creation_id, "thumb", identity, jobs, storage)


@router.delete("/v1/creations/{creation_id}", status_code=204, tags=["creations"])
async def delete_creation(
    creation_id: str,
    identity: Identity = Depends(resolve_identity),
    jobs: JobStore = Depends(get_jobs),
    storage: Storage = Depends(get_storage),
) -> Response:
    """Files first, then the record."""
    rec = await jobs.get_creation(identity.tenant_id, identity.user_id, creation_id)
    if rec is None:
        raise HTTPException(404, "creation not found")
    for key in (rec.image_key, rec.thumb_key):
        await storage.delete(identity.tenant_id, identity.user_id, key)
    await jobs.delete_creation(identity.tenant_id, identity.user_id, creation_id)
    return Response(status_code=204)
