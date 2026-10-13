"""Helpers shared by generation tests."""

from fastapi.testclient import TestClient
from wasifyai_art_api.runners import StubRunner
from wasifyai_art_api.worker import JobProcessor, NullLock


def versions(c: TestClient) -> dict:
    c.post("/v1/models/scan")
    return {m["slug"]: m["versions"][0]["id"] for m in c.get("/v1/models").json()["models"]}


def ready_klein(c: TestClient) -> str:
    vid = versions(c)["flux-2-klein-4b"]
    r = c.patch(f"/v1/model-versions/{vid}", json={"family": "flux2-klein", "status": "ready"})
    assert r.status_code == 200
    return vid


def processor(parts, runner=None, trainer=None) -> JobProcessor:
    return JobProcessor(
        parts["jobs"],
        parts["storage"],
        parts["bus"],
        runner or StubRunner(),
        NullLock(),
        5.0,
        parts["uploads"],
        trainer=trainer,
        training=parts["training"],
    )


def body(vid: str, **kw) -> dict:
    return {"provider_id": "comfyui-local", "version_id": vid, "prompt": "a red fox", **kw}


def ready_lora(c: TestClient, slug: str, family: str = "flux2-klein") -> str:
    vid = versions(c)[slug]
    r = c.patch(f"/v1/model-versions/{vid}", json={"family": family, "status": "ready"})
    assert r.status_code == 200
    return vid
