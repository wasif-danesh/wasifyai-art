import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from wasifyai_art_api.comfy import ComfyError
from wasifyai_art_api.main import create_app
from wasifyai_art_api.workflows import WorkflowRegistry

from tests.conftest import WORKFLOWS
from tests.support import body, processor, ready_klein, versions


async def test_full_flow_creates_a_creation(gen_client: TestClient, parts) -> None:
    vid = ready_klein(gen_client)
    r = gen_client.post("/v1/generations", json=body(vid, size="portrait", seed=42))
    assert r.status_code == 202
    job = r.json()
    assert job["status"] == "queued" and parts["bus"].queued == [job["id"]]

    await processor(parts).process(job["id"])

    done = gen_client.get(f"/v1/jobs/{job['id']}").json()
    assert done["status"] == "succeeded" and done["progress"] == 1.0
    c = gen_client.get(f"/v1/creations/{done['creation_id']}").json()
    assert (c["width"], c["height"]) == (864, 1152)
    assert c["params"]["seed"] == 42 and c["params"]["steps"] == 4
    assert c["models"][0]["name"] == "flux-2-klein-4b" and c["models"][0]["rel_path"].startswith(
        "diffusion_models/"
    )
    assert c["params"]["model_file"] == "flux-2-klein-4b.safetensors"
    png = gen_client.get(c["image_url"])
    assert png.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(png.content)).size == (864, 1152)
    thumb = gen_client.get(c["thumb_url"])
    assert thumb.headers["content-type"] == "image/jpeg"
    assert gen_client.get("/v1/creations").json()["creations"][0]["id"] == c["id"]


async def test_events_are_streamed_and_end_on_success(gen_client: TestClient, parts) -> None:
    vid = ready_klein(gen_client)
    job = gen_client.post("/v1/generations", json=body(vid)).json()
    await processor(parts).process(job["id"])
    with gen_client.stream("GET", f"/v1/jobs/{job['id']}/events") as r:
        text = "".join(r.iter_text())
    types = [line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("event: ")]
    assert types[0] == "queued" and "running" in types and "progress" in types
    assert types[-1] == "succeeded"
    last = [line for line in text.splitlines() if line.startswith("data: ")][-1]
    assert "creation_id" in json.loads(last[6:])


@pytest.mark.parametrize(
    ("patch", "status"),
    [
        ({"provider_id": "nope"}, 422),
        ({"size": "huge"}, 422),
        ({"version_id": "00000000-0000-0000-0000-000000000000"}, 404),
        ({"prompt": ""}, 422),
    ],
)
def test_refusals(gen_client: TestClient, patch: dict, status: int) -> None:
    vid = ready_klein(gen_client)
    assert gen_client.post("/v1/generations", json={**body(vid), **patch}).status_code == status


def test_unconfirmed_and_wrong_kind_are_refused(gen_client: TestClient) -> None:
    v = versions(gen_client)
    r = gen_client.post("/v1/generations", json=body(v["mystery"]))
    assert r.status_code == 409 and "not ready" in r.json()["detail"]
    r = gen_client.post("/v1/generations", json=body(v["some-lora"]))
    assert r.status_code == 422


def test_unconfigured_provider_is_refused(parts) -> None:
    settings = parts["settings"].model_copy(update={"comfyui_mode": "real", "comfyui_base_url": ""})
    c = TestClient(
        create_app(settings, parts["store"], parts["jobs"], parts["bus"], parts["storage"])
    )
    vid = ready_klein(c)
    r = c.post("/v1/generations", json=body(vid))
    assert r.status_code == 409 and "not configured" in r.json()["detail"]


async def test_failure_is_reported_safely(gen_client: TestClient, parts) -> None:
    class Boom:
        async def run(self, request, job_id, on_progress, inputs=None):
            raise ComfyError("execution_failed", "The generation failed in ComfyUI.")

    vid = ready_klein(gen_client)
    job = gen_client.post("/v1/generations", json=body(vid)).json()
    await processor(parts, Boom()).process(job["id"])
    got = gen_client.get(f"/v1/jobs/{job['id']}").json()
    assert got["status"] == "failed" and got["error"] == "The generation failed in ComfyUI."
    assert parts["jobs"].creations == []


async def test_unexpected_errors_do_not_leak_details(gen_client: TestClient, parts) -> None:
    class Bug:
        async def run(self, request, job_id, on_progress, inputs=None):
            raise RuntimeError("secret internal detail /Users/x")

    vid = ready_klein(gen_client)
    job = gen_client.post("/v1/generations", json=body(vid)).json()
    await processor(parts, Bug()).process(job["id"])
    assert gen_client.get(f"/v1/jobs/{job['id']}").json()["error"] == "The generation failed."


async def test_other_users_see_nothing_and_delete_removes_files(
    gen_client: TestClient, parts
) -> None:
    vid = ready_klein(gen_client)
    job = gen_client.post("/v1/generations", json=body(vid)).json()
    await processor(parts).process(job["id"])
    cid = gen_client.get(f"/v1/jobs/{job['id']}").json()["creation_id"]
    assert len(parts["storage"].objects) == 2

    app = gen_client.app
    app.state.settings = app.state.settings.model_copy(update={"stub_user_id": "other"})  # type: ignore[attr-defined]
    assert gen_client.get(f"/v1/creations/{cid}").status_code == 404
    assert gen_client.get(f"/v1/creations/{cid}/image").status_code == 404
    assert gen_client.get(f"/v1/jobs/{job['id']}").status_code == 404
    assert gen_client.get("/v1/creations").json()["creations"] == []
    app.state.settings = app.state.settings.model_copy(update={"stub_user_id": "dev"})  # type: ignore[attr-defined]

    assert gen_client.delete(f"/v1/creations/{cid}").status_code == 204
    assert parts["storage"].objects == {}
    assert gen_client.get(f"/v1/creations/{cid}").status_code == 404


def test_klein_workflow_map_is_consistent_and_fills() -> None:
    reg = WorkflowRegistry.load(WORKFLOWS)
    assert reg.problems() == []
    wf = reg.find("text2img", "flux2-klein")
    assert wf is not None
    g = wf.fill(
        {
            "model": "x.safetensors",
            "prompt": "p",
            "seed": 7,
            "steps": 4,
            "width": 864,
            "height": 1152,
        }
    )
    assert g["1"]["inputs"]["unet_name"] == "x.safetensors"
    assert g["9"]["inputs"]["noise_seed"] == 7
    assert g["6"]["inputs"]["width"] == g["7"]["inputs"]["width"] == 864
    assert reg.find("text2img", "sdxl") is None


def test_edit_workflow_map_is_consistent_and_fills() -> None:
    reg = WorkflowRegistry.load(WORKFLOWS)
    wf = reg.find("edit", "flux2-klein")
    assert wf is not None and wf.problems() == []
    g = wf.fill(
        {"model": "m.safetensors", "prompt": "p", "seed": 3, "steps": 4, "image": "a.png [temp]"}
    )
    assert g["6"]["inputs"]["image"] == "a.png [temp]" and g["15"]["inputs"]["noise_seed"] == 3
    assert "width" not in wf.map.inputs  # the picture decides the size
