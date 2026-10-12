from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from wasifyai_art_api.main import create_app
from wasifyai_art_api.settings import Settings
from wasifyai_art_api.store import InMemoryModelStore

from tests.conftest import FAMILIES, FOLDERS, PROVIDERS
from tests.helpers import write_safetensors


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    write_safetensors(root / "loras" / "kaia-style.safetensors",
                      {"modelspec.architecture": "flux-2-klein/lora", "modelspec.title": "Kaia Style"})  # fmt: skip
    write_safetensors(root / "diffusion_models" / "flux-2-klein-4b.safetensors")
    write_safetensors(root / "vae" / "ignored-folder.safetensors")  # not a hub folder
    (root / "loras" / "old.ckpt").write_bytes(b"pickle")
    (root / "loras" / "broken.safetensors").write_bytes(b"short")
    return root


@pytest.fixture
def scan_client(models_dir: Path, store: InMemoryModelStore) -> TestClient:
    settings = Settings(
        providers_file=PROVIDERS, families_file=FAMILIES, model_folders_file=FOLDERS,
        models_dir=models_dir,
    )  # fmt: skip
    return TestClient(create_app(settings, store))


def by_slug(client: TestClient) -> dict:
    return {m["slug"]: m for m in client.get("/v1/models").json()["models"]}


def test_scan_requires_models_dir(client: TestClient) -> None:
    assert client.post("/v1/models/scan").status_code == 409


def test_first_scan_registers_and_reports(scan_client: TestClient) -> None:
    r = scan_client.post("/v1/models/scan").json()
    assert (r["added"], r["unchanged"], r["missing"]) == (2, 0, 0)
    assert r["skipped_unsupported"] == ["loras/old.ckpt"]
    assert [p["rel_path"] for p in r["problems"]] == ["loras/broken.safetensors"]
    models = by_slug(scan_client)
    assert set(models) == {"kaia-style", "flux-2-klein-4b"}
    lora = models["kaia-style"]
    assert lora["kind"] == "lora" and models["flux-2-klein-4b"]["kind"] == "checkpoint"
    v = lora["versions"][0]
    assert v["status"] == "unconfirmed" and v["family"] is None
    assert v["family_hint"] == "flux2-klein"
    assert v["files"][0]["rel_path"] == "loras/kaia-style.safetensors"


def test_rescan_is_idempotent(scan_client: TestClient) -> None:
    scan_client.post("/v1/models/scan")
    r = scan_client.post("/v1/models/scan").json()
    assert (r["added"], r["updated"], r["unchanged"]) == (0, 0, 2)
    assert len(by_slug(scan_client)) == 2


def test_confirming_needs_a_family(scan_client: TestClient) -> None:
    scan_client.post("/v1/models/scan")
    vid = by_slug(scan_client)["kaia-style"]["versions"][0]["id"]
    r = scan_client.patch(f"/v1/model-versions/{vid}", json={"status": "ready"})
    assert r.status_code == 409
    r = scan_client.patch(f"/v1/model-versions/{vid}", json={"family": "nope"})
    assert r.status_code == 422
    r = scan_client.patch(
        f"/v1/model-versions/{vid}",
        json={"family": "flux2-klein", "trigger_words": [" kaia_x ", ""], "status": "ready"},
    )
    assert r.status_code == 200
    body = r.json()
    assert (body["family"], body["status"], body["trigger_words"]) == (
        "flux2-klein",
        "ready",
        ["kaia_x"],
    )


def test_missing_then_restored_file(scan_client: TestClient, models_dir: Path) -> None:
    scan_client.post("/v1/models/scan")
    vid = by_slug(scan_client)["kaia-style"]["versions"][0]["id"]
    scan_client.patch(
        f"/v1/model-versions/{vid}", json={"family": "flux2-klein", "status": "ready"}
    )
    path = models_dir / "loras" / "kaia-style.safetensors"
    saved = path.read_bytes()
    path.unlink()
    r = scan_client.post("/v1/models/scan").json()
    assert r["missing"] == 1
    v = by_slug(scan_client)["kaia-style"]["versions"][0]
    assert v["status"] == "missing" and v["files"][0]["status"] == "missing"
    assert (
        scan_client.patch(f"/v1/model-versions/{vid}", json={"status": "ready"}).status_code == 409
    )
    path.write_bytes(saved)
    r = scan_client.post("/v1/models/scan").json()
    assert r["restored"] == 1
    assert by_slug(scan_client)["kaia-style"]["versions"][0]["status"] == "ready"


def test_changed_file_needs_confirming_again(scan_client: TestClient, models_dir: Path) -> None:
    scan_client.post("/v1/models/scan")
    vid = by_slug(scan_client)["kaia-style"]["versions"][0]["id"]
    scan_client.patch(
        f"/v1/model-versions/{vid}", json={"family": "flux2-klein", "status": "ready"}
    )
    write_safetensors(models_dir / "loras" / "kaia-style.safetensors", tensors=5)
    r = scan_client.post("/v1/models/scan").json()
    assert r["updated"] == 1
    v = by_slug(scan_client)["kaia-style"]["versions"][0]
    assert v["status"] == "unconfirmed" and v["family"] == "flux2-klein"


def test_other_users_cannot_confirm(scan_client: TestClient, store: InMemoryModelStore) -> None:
    scan_client.post("/v1/models/scan")
    vid = by_slug(scan_client)["kaia-style"]["versions"][0]["id"]
    app = scan_client.app
    app.state.settings = app.state.settings.model_copy(update={"stub_user_id": "someone-else"})  # type: ignore[attr-defined]
    r = scan_client.patch(f"/v1/model-versions/{vid}", json={"family": "sdxl"})
    assert r.status_code == 404
