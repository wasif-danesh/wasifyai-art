from fastapi import APIRouter, Depends, HTTPException, Request
from wasifyai_art_contracts import (
    FamilyList,
    Health,
    ModelList,
    ModelVersionSummary,
    ProviderList,
    ScanResult,
    VersionUpdate,
)

from wasifyai_art_api import PRODUCT_ID
from wasifyai_art_api.families import FamilyConfig, to_info
from wasifyai_art_api.hub import scan_into_registry
from wasifyai_art_api.identity import Identity, resolve_identity
from wasifyai_art_api.limits import RateLimiter
from wasifyai_art_api.providers import ProviderConfig, describe
from wasifyai_art_api.settings import Settings, get_app_settings
from wasifyai_art_api.store import ModelStore, VersionConflict
from wasifyai_art_api.uploads import UploadStore

router = APIRouter()


def get_store(request: Request) -> ModelStore:
    return request.app.state.store


def get_providers(request: Request) -> list[ProviderConfig]:
    return request.app.state.providers


def get_uploads(request: Request) -> UploadStore:
    return request.app.state.uploads


def get_limiter(request: Request) -> RateLimiter:
    return request.app.state.limiter


def get_families(request: Request) -> list[FamilyConfig]:
    return request.app.state.families


@router.get("/health", response_model=Health, tags=["health"])
async def health() -> Health:
    return Health(product=PRODUCT_ID)


@router.get("/v1/providers", response_model=ProviderList, tags=["providers"])
async def list_providers(
    identity: Identity = Depends(resolve_identity),
    configs: list[ProviderConfig] = Depends(get_providers),
    settings: Settings = Depends(get_app_settings),
) -> ProviderList:
    env = {"COMFYUI_BASE_URL": settings.comfyui_base_url}
    return ProviderList(providers=describe(configs, env))


@router.get("/v1/families", response_model=FamilyList, tags=["models"])
async def list_families(
    identity: Identity = Depends(resolve_identity),
    families: list[FamilyConfig] = Depends(get_families),
) -> FamilyList:
    return FamilyList(families=to_info(families))


@router.get("/v1/models", response_model=ModelList, tags=["models"])
async def list_models(
    identity: Identity = Depends(resolve_identity), store: ModelStore = Depends(get_store)
) -> ModelList:
    return ModelList(models=await store.list_models(identity.tenant_id, identity.user_id))


@router.post("/v1/models/scan", response_model=ScanResult, tags=["models"])
async def scan_models(
    request: Request,
    identity: Identity = Depends(resolve_identity),
    store: ModelStore = Depends(get_store),
    settings: Settings = Depends(get_app_settings),
    families: list[FamilyConfig] = Depends(get_families),
) -> ScanResult:
    """Read the model folders' `.safetensors` headers and reconcile them with the registry."""
    if settings.models_dir is None:
        raise HTTPException(409, "MODELS_DIR is not set: point it at your ComfyUI models folder")
    try:
        return await scan_into_registry(
            store,
            identity.tenant_id,
            identity.user_id,
            settings.models_dir,
            request.app.state.folders,
            families,
        )
    except FileNotFoundError:
        raise HTTPException(409, "The models folder does not exist") from None


@router.patch(
    "/v1/model-versions/{version_id}", response_model=ModelVersionSummary, tags=["models"]
)
async def update_version(
    version_id: str,
    body: VersionUpdate,
    identity: Identity = Depends(resolve_identity),
    store: ModelStore = Depends(get_store),
    families: list[FamilyConfig] = Depends(get_families),
) -> ModelVersionSummary:
    if body.family is not None and body.family not in {f.id for f in families}:
        raise HTTPException(422, f"unknown family {body.family!r}")
    try:
        updated = await store.update_version(identity.tenant_id, identity.user_id, version_id, body)
    except VersionConflict as e:
        raise HTTPException(409, str(e)) from None
    if updated is None:  # also another user's version: no existence leak
        raise HTTPException(404, "model version not found")
    return updated
