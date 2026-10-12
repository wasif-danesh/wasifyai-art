"""Environment wiring and secrets, read only through pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from fastapi import Request
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://wsh:wsh@localhost:5433/wsh"
    redis_url: str = "redis://localhost:6380/0"
    providers_file: Path = REPO_ROOT / "config" / "providers.yaml"
    comfyui_base_url: str = ""
    # The ComfyUI `models` folder the hub scans (read-only). Unset: scanning answers 409.
    models_dir: Path | None = None
    model_folders_file: Path = REPO_ROOT / "config" / "model_folders.yaml"
    families_file: Path = REPO_ROOT / "config" / "families.yaml"

    # Generation. COMFYUI_MODE=stub makes placeholder images with simulated progress (no GPU).
    comfyui_mode: str = "stub"
    workflows_dir: Path = REPO_ROOT / "workflows"
    gpu_id: str = "gpu0"
    upload_max_bytes: int = 10 * 1024 * 1024
    upload_max_side: int = 2048
    lora_weight_min: float = -1.0
    lora_weight_max: float = 2.0
    job_timeout_s: float = 1200.0
    storage_backend: str = "local"  # local | s3
    data_dir: Path = REPO_ROOT / "data"
    storage_endpoint: str = "http://localhost:8334"
    storage_bucket: str = "wasifyai-art"
    storage_access_key: str = ""
    storage_secret_key: str = ""

    # Identity (CLAUDE.md rule 4): every request resolves a user. Only the dev stub exists so far.
    auth_mode: str = "stub"
    stub_tenant_id: str = "local"
    stub_user_id: str = "dev"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_app_settings(request: Request) -> Settings:
    """The settings the application was created with (tests build apps with their own)."""
    return request.app.state.settings
