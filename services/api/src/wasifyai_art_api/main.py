import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from wasifyai_art_api.bus import EventBus, InMemoryBus, RedisBus
from wasifyai_art_api.families import load_families
from wasifyai_art_api.jobs import InMemoryJobStore, JobStore, PostgresJobStore
from wasifyai_art_api.limits import InMemoryLimiter, RateLimiter, RedisLimiter
from wasifyai_art_api.providers import load_providers
from wasifyai_art_api.routes import router
from wasifyai_art_api.routes_generation import router as generation_router
from wasifyai_art_api.routes_uploads import router as uploads_router
from wasifyai_art_api.scanner import load_folder_map
from wasifyai_art_api.settings import Settings, get_settings
from wasifyai_art_api.storage import InMemoryStorage, Storage, make_storage
from wasifyai_art_api.store import InMemoryModelStore, ModelStore, PostgresModelStore
from wasifyai_art_api.uploads import (
    InMemoryUploadStore,
    PostgresUploadStore,
    UploadStore,
    sweep_uploads,
)
from wasifyai_art_api.workflows import WorkflowRegistry


def create_app(
    settings: Settings | None = None,
    store: ModelStore | None = None,
    jobs: JobStore | None = None,
    bus: EventBus | None = None,
    storage: Storage | None = None,
    uploads: UploadStore | None = None,
    limiter: RateLimiter | None = None,
) -> FastAPI:
    """Production wiring when no parts are given; tests pass in-memory parts."""
    settings = settings or get_settings()
    if settings.auth_mode != "stub":
        raise RuntimeError("AUTH_MODE must be 'stub': no other identity mode exists yet")
    engine: AsyncEngine | None = None
    redis: Redis | None = None
    production = store is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        sweeper = asyncio.create_task(_sweep_forever(app)) if production else None
        yield
        if sweeper is not None:
            sweeper.cancel()
        if engine is not None:
            await engine.dispose()
        if redis is not None:
            await redis.aclose()

    app = FastAPI(title="wasifyai-art API", lifespan=lifespan)
    app.state.settings = settings
    app.state.providers = load_providers(settings.providers_file)
    app.state.families = load_families(settings.families_file)
    app.state.folders = load_folder_map(settings.model_folders_file)
    workflows = WorkflowRegistry.load(settings.workflows_dir)
    if problems := workflows.problems():
        raise RuntimeError("invalid workflows:\n  - " + "\n  - ".join(problems))
    app.state.workflows = workflows
    if production:
        engine = create_async_engine(settings.database_url)
        redis = Redis.from_url(settings.redis_url, socket_timeout=30)  # > SSE block (10 s)
        store = PostgresModelStore(engine)
        jobs = jobs or PostgresJobStore(engine)
        bus = bus or RedisBus(redis)
        storage = storage or make_storage(settings)
        uploads = uploads or PostgresUploadStore(engine)
        limiter = limiter or RedisLimiter(redis)
    app.state.store = store or InMemoryModelStore()
    app.state.jobs = jobs or InMemoryJobStore()
    app.state.bus = bus or InMemoryBus()
    app.state.storage = storage or InMemoryStorage()
    app.include_router(router)
    app.state.uploads = uploads or InMemoryUploadStore()
    app.state.limiter = limiter or InMemoryLimiter()
    app.include_router(generation_router)
    app.include_router(uploads_router)
    return app


async def _sweep_forever(app: FastAPI) -> None:
    """Hourly: delete pictures that were uploaded and never used."""
    log = logging.getLogger("wasifyai_art_api")
    while True:
        try:
            n = await sweep_uploads(app.state.uploads, app.state.storage)
            if n:
                log.info("swept %d unused uploads", n)
        except Exception:
            log.exception("upload sweep failed")
        await asyncio.sleep(3600)
