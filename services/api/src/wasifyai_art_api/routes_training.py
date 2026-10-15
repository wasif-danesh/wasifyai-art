"""Datasets, training runs and publishing a trained LoRA into the hub."""

import asyncio
import io
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from PIL import Image
from wasifyai_art_contracts import (
    CaptionUpdate,
    CheckpointInfo,
    DatasetCreate,
    DatasetImageSummary,
    DatasetList,
    DatasetSummary,
    ModelVersionSummary,
    PublishRequest,
    TrainingList,
    TrainingRequest,
    TrainingSummary,
    VersionUpdate,
)

from wasifyai_art_api.bus import EventBus
from wasifyai_art_api.families import FamilyConfig
from wasifyai_art_api.hub import scan_into_registry
from wasifyai_art_api.identity import Identity, resolve_identity
from wasifyai_art_api.jobs import JobStore
from wasifyai_art_api.limits import RateLimiter
from wasifyai_art_api.lora_convert import NotSafetensors, to_comfyui
from wasifyai_art_api.routes import get_families, get_limiter, get_store
from wasifyai_art_api.routes_generation import get_bus, get_jobs, get_storage
from wasifyai_art_api.settings import Settings, get_app_settings
from wasifyai_art_api.storage import Storage
from wasifyai_art_api.store import ModelStore, slugify
from wasifyai_art_api.training import (
    DatasetRecord,
    FamilyTraining,
    ImageRecord,
    RunRecord,
    TrainingStore,
)
from wasifyai_art_api.uploads import UploadRefused, sanitize

router = APIRouter()
FAMILY = "flux2-klein"
THUMB_PX = 480


def get_training(request: Request) -> TrainingStore:
    return request.app.state.training


def get_training_cfg(request: Request) -> dict[str, FamilyTraining]:
    return request.app.state.training_cfg


def _dataset_summary(d: DatasetRecord, images: list[ImageRecord]) -> DatasetSummary:
    return DatasetSummary(
        id=d.id, name=d.name, trigger_word=d.trigger_word, image_count=len(images),
        created_at=d.created_at,
        images=[
            DatasetImageSummary(
                id=i.id, width=i.width, height=i.height, caption=i.caption,
                has_trigger=d.trigger_word.lower() in i.caption.lower(),
                thumb_url=f"/v1/datasets/{d.id}/images/{i.id}/thumb",
            )
            for i in images
        ],
    )  # fmt: skip


async def _owned_dataset(
    dataset_id: str, identity: Identity, training: TrainingStore
) -> DatasetRecord:
    d = await training.get_dataset(identity.tenant_id, identity.user_id, dataset_id)
    if d is None:
        raise HTTPException(404, "dataset not found")
    return d


@router.post("/v1/datasets", response_model=DatasetSummary, status_code=201, tags=["training"])
async def create_dataset(
    body: DatasetCreate,
    identity: Identity = Depends(resolve_identity),
    training: TrainingStore = Depends(get_training),
) -> DatasetSummary:
    d = await training.create_dataset(
        identity.tenant_id, identity.user_id, body.name.strip(), body.trigger_word
    )
    return _dataset_summary(d, [])


@router.get("/v1/datasets", response_model=DatasetList, tags=["training"])
async def list_datasets(
    identity: Identity = Depends(resolve_identity),
    training: TrainingStore = Depends(get_training),
) -> DatasetList:
    out = []
    for d in await training.list_datasets(identity.tenant_id, identity.user_id):
        out.append(_dataset_summary(d, await training.list_images(d.id)))
    return DatasetList(datasets=out)


@router.get("/v1/datasets/{dataset_id}", response_model=DatasetSummary, tags=["training"])
async def get_dataset(
    dataset_id: str,
    identity: Identity = Depends(resolve_identity),
    training: TrainingStore = Depends(get_training),
) -> DatasetSummary:
    d = await _owned_dataset(dataset_id, identity, training)
    return _dataset_summary(d, await training.list_images(d.id))


@router.delete("/v1/datasets/{dataset_id}", status_code=204, tags=["training"])
async def delete_dataset(
    dataset_id: str,
    identity: Identity = Depends(resolve_identity),
    training: TrainingStore = Depends(get_training),
    storage: Storage = Depends(get_storage),
) -> Response:
    images = await training.delete_dataset(identity.tenant_id, identity.user_id, dataset_id)
    if images is None:
        raise HTTPException(404, "dataset not found")
    for i in images:
        for key in (i.key, i.thumb_key):
            await storage.delete(identity.tenant_id, identity.user_id, key)
    return Response(status_code=204)


def _thumb(png: bytes) -> bytes:
    with Image.open(io.BytesIO(png)) as im:
        rgb = im.convert("RGB")
    rgb.thumbnail((THUMB_PX, THUMB_PX))
    out = io.BytesIO()
    rgb.save(out, "JPEG", quality=82)
    return out.getvalue()


@router.post(
    "/v1/datasets/{dataset_id}/images",
    response_model=DatasetImageSummary,
    status_code=201,
    tags=["training"],
)
async def add_image(
    dataset_id: str,
    request: Request,
    identity: Identity = Depends(resolve_identity),
    training: TrainingStore = Depends(get_training),
    storage: Storage = Depends(get_storage),
    limiter: RateLimiter = Depends(get_limiter),
    settings: Settings = Depends(get_app_settings),
    cfg: dict[str, FamilyTraining] = Depends(get_training_cfg),
) -> DatasetImageSummary:
    """The raw picture is the request body, re-encoded and stripped of metadata like any upload.

    Its caption starts as the trigger word."""
    d = await _owned_dataset(dataset_id, identity, training)
    existing = await training.list_images(d.id)
    if len(existing) >= cfg[FAMILY].max_images:
        raise HTTPException(409, f"A dataset holds at most {cfg[FAMILY].max_images} pictures.")
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > settings.upload_max_bytes:
        raise HTTPException(413, "That picture is too large (over 10 MB).")
    if not (request.headers.get("content-type") or "").startswith("image/"):
        raise HTTPException(415, "Send the picture as an image/* request body.")
    if not await limiter.hit(
        f"dataset:{identity.tenant_id}:{identity.user_id}", settings.dataset_upload_per_hour, 3600
    ):
        raise HTTPException(429, "Too many uploads. Try again later.")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > settings.upload_max_bytes:
            raise HTTPException(413, "That picture is too large (over 10 MB).")
        chunks.append(chunk)
    try:
        png, w, h = await asyncio.to_thread(sanitize, b"".join(chunks), settings.upload_max_side)
    except UploadRefused as e:
        raise HTTPException(e.status, e.message) from None
    thumb = await asyncio.to_thread(_thumb, png)
    from uuid import uuid4  # noqa: PLC0415

    iid = str(uuid4())
    key, tkey = f"datasets/{d.id}/{iid}.png", f"datasets/{d.id}/{iid}.thumb.jpg"
    await storage.put(identity.tenant_id, identity.user_id, key, png)
    await storage.put(identity.tenant_id, identity.user_id, tkey, thumb)
    rec = ImageRecord(iid, d.id, key, tkey, w, h, d.trigger_word)
    await training.add_image(d.id, rec)
    return _dataset_summary(d, [rec]).images[0]


@router.patch(
    "/v1/datasets/{dataset_id}/images/{image_id}",
    response_model=DatasetImageSummary,
    tags=["training"],
)
async def update_caption(
    dataset_id: str,
    image_id: str,
    body: CaptionUpdate,
    identity: Identity = Depends(resolve_identity),
    training: TrainingStore = Depends(get_training),
) -> DatasetImageSummary:
    d = await _owned_dataset(dataset_id, identity, training)
    if await training.get_image(d.id, image_id) is None:
        raise HTTPException(404, "picture not found")
    await training.set_caption(d.id, image_id, body.caption.strip())
    image = await training.get_image(d.id, image_id)
    assert image is not None
    return _dataset_summary(d, [image]).images[0]


@router.delete("/v1/datasets/{dataset_id}/images/{image_id}", status_code=204, tags=["training"])
async def delete_image(
    dataset_id: str,
    image_id: str,
    identity: Identity = Depends(resolve_identity),
    training: TrainingStore = Depends(get_training),
    storage: Storage = Depends(get_storage),
) -> Response:
    d = await _owned_dataset(dataset_id, identity, training)
    image = await training.get_image(d.id, image_id)
    if image is None:
        raise HTTPException(404, "picture not found")
    for key in (image.key, image.thumb_key):
        await storage.delete(identity.tenant_id, identity.user_id, key)
    await training.delete_image(d.id, image_id)
    return Response(status_code=204)


@router.get("/v1/datasets/{dataset_id}/images/{image_id}/thumb", tags=["training"])
async def image_thumb(
    dataset_id: str,
    image_id: str,
    identity: Identity = Depends(resolve_identity),
    training: TrainingStore = Depends(get_training),
    storage: Storage = Depends(get_storage),
) -> Response:
    d = await _owned_dataset(dataset_id, identity, training)
    image = await training.get_image(d.id, image_id)
    if image is None:
        raise HTTPException(404, "picture not found")
    try:
        data = await storage.get(identity.tenant_id, identity.user_id, image.thumb_key)
    except FileNotFoundError:
        raise HTTPException(404, "file not found") from None
    return Response(
        data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"}
    )


# ---- training runs ----------------------------------------------------------------------------


def _summary(run: RunRecord, job: Any, image_count: int) -> TrainingSummary:
    return TrainingSummary(
        id=run.job_id, dataset_id=run.dataset_id or "", dataset_name=run.name,
        trigger_word=run.trigger_word, status=job.status if job else "failed",
        progress=job.progress if job else 0.0, error=job.error if job else None,
        preset=run.preset, steps=int(run.params.get("steps", 0)), image_count=image_count,
        created_at=run.created_at,
        checkpoints=[
            CheckpointInfo(
                step=c["step"], size_bytes=c["size_bytes"],
                preview_url=(
                    f"/v1/trainings/{run.job_id}/checkpoints/{c['step']}/preview"
                    if c.get("preview_key")
                    else None
                ),
            )
            for c in run.checkpoints
        ],
    )  # fmt: skip


@router.post("/v1/trainings", response_model=TrainingSummary, status_code=202, tags=["training"])
async def start_training(
    body: TrainingRequest,
    identity: Identity = Depends(resolve_identity),
    training: TrainingStore = Depends(get_training),
    jobs: JobStore = Depends(get_jobs),
    bus: EventBus = Depends(get_bus),
    cfg: dict[str, FamilyTraining] = Depends(get_training_cfg),
    settings: Settings = Depends(get_app_settings),
) -> TrainingSummary:
    """Validate a dataset and queue a training job. GPU work is never awaited here."""
    d = await _owned_dataset(body.dataset_id, identity, training)
    fam = cfg[FAMILY]
    images = await training.list_images(d.id)
    if len(images) < fam.min_images:
        raise HTTPException(409, f"Add at least {fam.min_images} pictures before training.")
    if settings.trainer_mode == "real" and not await asyncio.to_thread(settings.mflux_bin.is_file):
        raise HTTPException(409, "The trainer (mflux) is not installed on this machine.")
    preset = fam.presets[body.preset]
    epochs = body.epochs or int(preset["epochs"])
    steps = len(images) * epochs
    save_every = max(25, round(steps / 6 / 25) * 25)
    params = {**fam.defaults, "steps": steps}
    if body.rank:
        params["rank"] = body.rank
    request = {
        "kind": "train", "dataset_id": d.id, "name": d.name, "trigger_word": d.trigger_word,
        "family": FAMILY, "engine": fam.engine, "model": fam.model, "preset": body.preset,
        "epochs": epochs, "steps": steps, "save_every": save_every, "params": params,
        "image_count": len(images),
        # shown in the jobs list
        "prompt": f"Train LoRA: {d.name}", "mode": "train", "model_name": fam.model,
    }  # fmt: skip
    job = await jobs.create_job(identity.tenant_id, identity.user_id, request, kind="train")
    run = RunRecord(job.id, d.id, identity.tenant_id, identity.user_id, d.name, d.trigger_word,
                    FAMILY, body.preset, params)  # fmt: skip
    await training.create_run(run)
    run.created_at = run.created_at or job.created_at
    await bus.publish(job.id, "queued", {})
    await bus.enqueue(job.id)  # type: ignore[attr-defined]
    return _summary(run, job, len(images))


@router.get("/v1/trainings", response_model=TrainingList, tags=["training"])
async def list_trainings(
    dataset_id: str | None = None,
    identity: Identity = Depends(resolve_identity),
    training: TrainingStore = Depends(get_training),
    jobs: JobStore = Depends(get_jobs),
) -> TrainingList:
    out = []
    for run in await training.list_runs(identity.tenant_id, identity.user_id):
        if dataset_id and run.dataset_id != dataset_id:
            continue
        job = await jobs.get_job(identity.tenant_id, identity.user_id, run.job_id)
        out.append(_summary(run, job, int(run.params.get("image_count", 0)) or _count(job)))
    return TrainingList(trainings=out)


def _count(job: Any) -> int:
    return int(job.request.get("image_count", 0)) if job else 0


async def _run_for(training_id: str, identity: Identity, training: TrainingStore) -> RunRecord:
    run = await training.get_run(identity.tenant_id, identity.user_id, training_id)
    if run is None:
        raise HTTPException(404, "training not found")
    return run


@router.get("/v1/trainings/{training_id}", response_model=TrainingSummary, tags=["training"])
async def get_training_run(
    training_id: str,
    identity: Identity = Depends(resolve_identity),
    training: TrainingStore = Depends(get_training),
    jobs: JobStore = Depends(get_jobs),
) -> TrainingSummary:
    run = await _run_for(training_id, identity, training)
    job = await jobs.get_job(identity.tenant_id, identity.user_id, run.job_id)
    return _summary(run, job, _count(job))


@router.get("/v1/trainings/{training_id}/checkpoints/{step}/preview", tags=["training"])
async def checkpoint_preview(
    training_id: str,
    step: int,
    identity: Identity = Depends(resolve_identity),
    training: TrainingStore = Depends(get_training),
    storage: Storage = Depends(get_storage),
) -> Response:
    run = await _run_for(training_id, identity, training)
    cp = next((c for c in run.checkpoints if c["step"] == step), None)
    if cp is None or not cp.get("preview_key"):
        raise HTTPException(404, "preview not found")
    try:
        data = await storage.get(identity.tenant_id, identity.user_id, cp["preview_key"])
    except FileNotFoundError:
        raise HTTPException(404, "file not found") from None
    return Response(
        data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"}
    )


@router.post(
    "/v1/trainings/{training_id}/publish",
    response_model=ModelVersionSummary,
    status_code=201,
    tags=["training"],
)
async def publish_checkpoint(
    training_id: str,
    body: PublishRequest,
    request: Request,
    identity: Identity = Depends(resolve_identity),
    training: TrainingStore = Depends(get_training),
    storage: Storage = Depends(get_storage),
    store: ModelStore = Depends(get_store),
    settings: Settings = Depends(get_app_settings),
    families: list[FamilyConfig] = Depends(get_families),
) -> ModelVersionSummary:
    """Convert a checkpoint for ComfyUI, place it in the models folder, register it as ready."""
    run = await _run_for(training_id, identity, training)
    cp = next((c for c in run.checkpoints if c["step"] == body.step), None)
    if cp is None:
        raise HTTPException(404, "checkpoint not found")
    if settings.models_dir is None:
        raise HTTPException(409, "MODELS_DIR is not set: point it at your ComfyUI models folder")
    raw = await storage.get(identity.tenant_id, identity.user_id, cp["adapter_key"])
    title = (body.name or run.name).strip() or run.name
    try:
        data = await asyncio.to_thread(to_comfyui, raw, title=title, trigger_word=run.trigger_word)
    except NotSafetensors:
        raise HTTPException(422, "The checkpoint is not a valid LoRA file.") from None
    folder = settings.models_dir / "loras"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{slugify(title)}-step{body.step}"
    path = folder / f"{stem}.safetensors"
    n = 2
    while path.exists():
        path, n = folder / f"{stem}-{n}.safetensors", n + 1
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.rename(path)  # atomic: a scan never sees a half-written file
    await scan_into_registry(
        store, identity.tenant_id, identity.user_id, settings.models_dir,
        request.app.state.folders, families,
    )  # fmt: skip
    rel = f"loras/{path.name}"
    for m in await store.list_models(identity.tenant_id, identity.user_id):
        for v in m.versions:
            if any(f.rel_path == rel for f in v.files):
                updated = await store.update_version(
                    identity.tenant_id, identity.user_id, v.id,
                    VersionUpdate(family=FAMILY, trigger_words=[run.trigger_word], status="ready"),
                )  # fmt: skip
                assert updated is not None
                return updated
    raise HTTPException(500, "The LoRA was written but could not be registered.")
