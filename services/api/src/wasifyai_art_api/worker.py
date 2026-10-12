"""The job worker: `python -m wasifyai_art_api.worker`. One job at a time per GPU id.

Runs as its own process, separate from the API, so GPU work and long ComfyUI connections never block
requests. It lives in this package for now and can be split out without code changes."""

import asyncio
import contextlib
import logging
import os
import socket
import time
from collections.abc import AsyncIterator
from typing import Any, Protocol
from uuid import uuid4

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import create_async_engine
from wasifyai_art_contracts import CreationModelRef

from wasifyai_art_api.bus import GROUP, JOBS_STREAM, TERMINAL, EventBus, RedisBus
from wasifyai_art_api.comfy import ComfyClient, ComfyError
from wasifyai_art_api.images import reencode
from wasifyai_art_api.jobs import CreationRecord, JobStore, PostgresJobStore
from wasifyai_art_api.runners import ComfyRunner, Runner, StubRunner
from wasifyai_art_api.settings import Settings, get_settings
from wasifyai_art_api.storage import Storage, make_storage
from wasifyai_art_api.uploads import PostgresUploadStore, UploadStore, release
from wasifyai_art_api.workflows import WorkflowRegistry

log = logging.getLogger("wasifyai_art_worker")
PROGRESS_STEP = 0.05


class GpuLock(Protocol):
    def hold(self) -> contextlib.AbstractAsyncContextManager[None]: ...


class NullLock:
    @contextlib.asynccontextmanager
    async def hold(self) -> AsyncIterator[None]:
        yield


class RedisGpuLock:
    """One job at a time per GPU id across every worker that shares this Redis."""

    TTL_MS = 60_000

    def __init__(self, redis: Redis, gpu_id: str):
        self._r = redis
        self._key = f"wsh:gpu:{gpu_id}"

    @contextlib.asynccontextmanager
    async def hold(self) -> AsyncIterator[None]:
        token = uuid4().hex
        got = await self._r.set(self._key, token, nx=True, px=self.TTL_MS)
        while not got:  # another process holds the GPU; Redis has no blocking wait for a key
            await asyncio.sleep(1.0)  # noqa: ASYNC110
            got = await self._r.set(self._key, token, nx=True, px=self.TTL_MS)

        async def renew() -> None:
            while True:
                await asyncio.sleep(self.TTL_MS / 3000)
                if await self._r.get(self._key) == token.encode():
                    await self._r.pexpire(self._key, self.TTL_MS)

        task = asyncio.create_task(renew())
        try:
            yield
        finally:
            task.cancel()
            if await self._r.get(self._key) == token.encode():
                await self._r.delete(self._key)


class JobProcessor:
    def __init__(
        self, jobs: JobStore, storage: Storage, bus: EventBus, runner: Runner, lock: GpuLock,
        timeout_s: float = 1200.0, uploads: UploadStore | None = None,
    ):  # fmt: skip
        self.jobs, self.storage, self.bus = jobs, storage, bus
        self.runner, self.lock, self.timeout_s = runner, lock, timeout_s
        self.uploads = uploads

    async def _fail(self, job_id: str, message: str, gpu_seconds: float = 0.0) -> None:
        await self.jobs.finish_job(job_id, "failed", message, None, gpu_seconds)
        await self.bus.publish(job_id, "failed", {"message": message})

    async def _watch_cancel(self, job_id: str) -> None:
        """While the job runs, ask the provider to stop as soon as a cancel is requested."""
        while True:
            await asyncio.sleep(0.5)
            if await self.bus.cancel_requested(job_id):
                await self.runner.cancel(job_id)
                return

    async def process(self, job_id: str) -> None:
        try:
            await self._process(job_id)
        finally:
            await self._release_upload(job_id)

    async def _release_upload(self, job_id: str) -> None:
        """An uploaded input picture is not kept: once the job is over, it is deleted."""
        job = await self.jobs.get_job_any(job_id)
        if job is None or job.status not in TERMINAL or self.uploads is None:
            return
        upload_id = job.request.get("source_upload_id")
        if not upload_id:
            return
        rec = await self.uploads.get(job.tenant_id, job.owner_id, upload_id)
        if rec is not None:
            await release(self.uploads, self.storage, rec)

    async def _process(self, job_id: str) -> None:
        rec = await self.jobs.start_job(job_id)
        if rec is None:
            stale = await self.jobs.get_job_any(job_id)
            if stale and stale.status in ("queued", "running"):
                await self._fail(job_id, "The generation failed too many times.")
            return  # already finished: a redelivered message
        if await self.bus.cancel_requested(job_id):
            await self.jobs.finish_job(job_id, "canceled", None, None, 0.0)
            await self.bus.publish(job_id, "canceled", {})
            return
        await self.bus.publish(job_id, "running", {})

        last = 0.0

        async def on_progress(p: float) -> None:
            nonlocal last
            if p >= 1.0 or p - last >= PROGRESS_STEP:
                last = p
                await self.jobs.set_progress(job_id, round(p, 3))
                await self.bus.publish(job_id, "progress", {"progress": round(p, 3)})

        started = time.monotonic()
        inputs: dict[str, bytes] = {}
        if key := rec.request.get("source_key"):
            try:
                if not str(key).startswith(("creations/", "uploads/")):
                    raise FileNotFoundError(key)
                inputs["image"] = await self.storage.get(rec.tenant_id, rec.owner_id, key)
            except FileNotFoundError:
                await self._fail(job_id, "The source picture could not be found.")
                return
        watcher: asyncio.Task[None] | None = None
        try:
            async with self.lock.hold():
                if await self.bus.cancel_requested(job_id):
                    raise ComfyError("interrupted", "canceled before it started")
                started = time.monotonic()
                watcher = asyncio.create_task(self._watch_cancel(job_id))
                async with asyncio.timeout(self.timeout_s):
                    data = await self.runner.run(rec.request, job_id, on_progress, inputs or None)
            elapsed = time.monotonic() - started
            creation_id = await self._save(rec.tenant_id, rec.owner_id, rec.request, data)
        except ComfyError as e:
            if e.code == "interrupted" and await self.bus.cancel_requested(job_id):
                await self.jobs.finish_job(job_id, "canceled", None, None, 0.0)
                await self.bus.publish(job_id, "canceled", {})
                return
            await self._fail(job_id, e.message, time.monotonic() - started)
            return
        except TimeoutError:
            await self._fail(job_id, "The generation took too long.", time.monotonic() - started)
            return
        except Exception:
            log.exception("job %s failed unexpectedly", job_id)
            await self._fail(job_id, "The generation failed.", time.monotonic() - started)
            return
        finally:
            if watcher is not None:
                watcher.cancel()
        await self.jobs.finish_job(job_id, "succeeded", None, creation_id, round(elapsed, 2))
        await self.bus.publish(job_id, "succeeded", {"creation_id": creation_id})

    async def _save(self, tenant_id: str, owner_id: str, req: dict, data: bytes) -> str:
        png, width, height, thumb = await asyncio.to_thread(reencode, data)
        creation_id = str(uuid4())
        image_key = f"creations/{creation_id}/image.png"
        thumb_key = f"creations/{creation_id}/thumb.jpg"
        await self.storage.put(tenant_id, owner_id, image_key, png)
        await self.storage.put(tenant_id, owner_id, thumb_key, thumb)
        try:
            await self.jobs.add_creation(
                tenant_id,
                owner_id,
                CreationRecord(
                    id=creation_id,
                    prompt=req["prompt"],
                    params={
                        "family": req["family"],
                        "seed": req["seed"],
                        "steps": req["steps"],
                        "size": req["size"],
                        "workflow": req["workflow"],
                        "model_file": req["model_file"],
                        "mode": req.get("mode", "text"),
                        "source": "upload" if req.get("source_upload_id") else "creation",
                        "lora_count": len(req.get("loras", [])),
                    },  # fmt: skip
                    provider_id=req["provider_id"],
                    image_key=image_key,
                    thumb_key=thumb_key,
                    width=width,
                    height=height,
                    created_at="",
                    parent_id=req.get("parent_id"),
                    models=[
                        CreationModelRef(
                            version_id=req["version_id"],
                            name=req["model_name"],
                            family=req["family"],
                            rel_path=req["rel_path"],
                            header_sha256=req["header_sha256"],
                        ),
                        *[
                            CreationModelRef(
                                version_id=x["version_id"],
                                role="lora",
                                weight=x["weight"],
                                name=x["name"],
                                family=req["family"],
                                rel_path=x["rel_path"],
                                header_sha256=x["header_sha256"],
                            )
                            for x in req.get("loras", [])
                        ],
                    ],
                ),
            )
        except Exception:
            await self.storage.delete(tenant_id, owner_id, image_key)
            await self.storage.delete(tenant_id, owner_id, thumb_key)
            raise
        return creation_id


async def run_worker(settings: Settings) -> None:
    engine = create_async_engine(settings.database_url)
    # socket_timeout must exceed the longest blocking read (XREADGROUP BLOCK 5 s)
    redis = Redis.from_url(settings.redis_url, socket_timeout=30)
    bus = RedisBus(redis)
    await bus.ensure_group()
    workflows = WorkflowRegistry.load(settings.workflows_dir)
    runner: Runner = (
        ComfyRunner(ComfyClient(settings.comfyui_base_url), workflows, settings.job_timeout_s)
        if settings.comfyui_mode == "real"
        else StubRunner()
    )
    proc = JobProcessor(
        PostgresJobStore(engine), make_storage(settings), bus, runner,
        RedisGpuLock(redis, settings.gpu_id), settings.job_timeout_s, PostgresUploadStore(engine),
    )  # fmt: skip
    consumer = f"{socket.gethostname()}-{os.getpid()}"
    log.info("worker %s ready (mode=%s, gpu=%s)", consumer, settings.comfyui_mode, settings.gpu_id)
    idle_ms = int((settings.job_timeout_s + 120) * 1000)
    while True:
        # Take over jobs a crashed worker left unacknowledged, then wait for new ones.
        auto: Any = await redis.xautoclaim(JOBS_STREAM, GROUP, consumer, idle_ms, "0-0", count=1)
        claimed = auto[1]
        batch: Any = [(JOBS_STREAM, claimed)] if claimed else []
        if not batch:
            batch = await redis.xreadgroup(GROUP, consumer, {JOBS_STREAM: ">"}, count=1, block=5000)
        for _stream, entries in batch or []:
            for entry_id, fields in entries:
                job_id = fields.get(b"job_id", b"").decode()
                try:
                    await proc.process(job_id)
                finally:
                    await redis.xack(JOBS_STREAM, GROUP, entry_id)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(run_worker(get_settings()))


if __name__ == "__main__":
    main()
