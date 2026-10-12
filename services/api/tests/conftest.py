from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from wasifyai_art_api.main import create_app
from wasifyai_art_api.settings import Settings
from wasifyai_art_api.store import InMemoryModelStore

CONFIG = Path(__file__).resolve().parents[3] / "config"
PROVIDERS = CONFIG / "providers.yaml"
FAMILIES = CONFIG / "families.yaml"
FOLDERS = CONFIG / "model_folders.yaml"
WORKFLOWS = Path(__file__).resolve().parents[3] / "workflows"


@pytest.fixture
def providers_file() -> Path:
    return PROVIDERS


@pytest.fixture
def store() -> InMemoryModelStore:
    return InMemoryModelStore()


@pytest.fixture
def client(store: InMemoryModelStore) -> TestClient:
    settings = Settings(providers_file=PROVIDERS, workflows_dir=WORKFLOWS, comfyui_base_url="")
    return TestClient(create_app(settings, store))


from wasifyai_art_api.bus import InMemoryBus  # noqa: E402
from wasifyai_art_api.jobs import InMemoryJobStore  # noqa: E402
from wasifyai_art_api.storage import InMemoryStorage  # noqa: E402
from wasifyai_art_api.uploads import InMemoryUploadStore  # noqa: E402

from tests.helpers import write_safetensors  # noqa: E402


@pytest.fixture
def parts(tmp_path: Path):
    root = tmp_path / "models"
    write_safetensors(root / "diffusion_models" / "flux-2-klein-4b.safetensors")
    write_safetensors(root / "loras" / "some-lora.safetensors")
    write_safetensors(root / "diffusion_models" / "mystery.safetensors")
    meta = {"modelspec.architecture": "flux-2-klein/lora"}
    for name in ("kaia", "lora-b", "lora-c", "lora-d", "lora-e", "lora-f"):
        write_safetensors(root / "loras" / f"{name}.safetensors", meta)
    write_safetensors(root / "loras" / "sdxl-thing.safetensors")
    settings = Settings(
        providers_file=PROVIDERS, families_file=FAMILIES, model_folders_file=FOLDERS,
        workflows_dir=WORKFLOWS, models_dir=root, comfyui_mode="stub",
    )  # fmt: skip
    return {
        "settings": settings,
        "store": InMemoryModelStore(),
        "jobs": InMemoryJobStore(),
        "bus": InMemoryBus(),
        "storage": InMemoryStorage(),
        "uploads": InMemoryUploadStore(),
    }


@pytest.fixture
def gen_client(parts) -> TestClient:
    return TestClient(
        create_app(
            parts["settings"],
            parts["store"],
            parts["jobs"],
            parts["bus"],
            parts["storage"],
            parts["uploads"],
        )
    )
