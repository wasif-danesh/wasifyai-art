from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from wasifyai_art_api.main import create_app
from wasifyai_art_api.settings import Settings
from wasifyai_art_api.store import InMemoryModelStore
from wasifyai_art_contracts import ModelSummary, ModelVersionSummary


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "product": "wasifyai-art"}


def test_providers_are_unconfigured_without_an_address(client: TestClient) -> None:
    body = client.get("/v1/providers").json()
    local = next(p for p in body["providers"] if p["id"] == "comfyui-local")
    assert local["status"] == "unconfigured"
    assert local["capabilities"]["loras"] is True


def test_providers_are_configured_with_an_address(
    store: InMemoryModelStore, providers_file: Path
) -> None:
    settings = Settings(providers_file=providers_file, comfyui_base_url="http://x:8188")
    app = create_app(settings, store)
    body = TestClient(app).get("/v1/providers").json()
    assert body["providers"][0]["status"] == "configured"


def test_models_are_scoped_to_the_caller(client: TestClient, store: InMemoryModelStore) -> None:
    mine = ModelSummary(
        id="1", kind="lora", name="Mine", slug="mine",
        versions=[ModelVersionSummary(id="v1", label="v1", status="ready", trigger_words=["kaia"])],
    )  # fmt: skip
    theirs = ModelSummary(id="2", kind="lora", name="Theirs", slug="theirs")
    store.add_summary("local", "dev", mine)
    store.add_summary("local", "someone-else", theirs)
    store.add_summary("other", "dev", theirs)
    body = client.get("/v1/models").json()
    assert [m["slug"] for m in body["models"]] == ["mine"]
    assert body["models"][0]["versions"][0]["trigger_words"] == ["kaia"]


def test_only_the_stub_identity_mode_exists(store: InMemoryModelStore) -> None:
    with pytest.raises(RuntimeError):
        create_app(Settings(auth_mode="jwt"), store)
