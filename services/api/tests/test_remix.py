import io

from fastapi.testclient import TestClient
from PIL import Image
from wasifyai_art_api.runners import StubRunner

from tests.support import body, processor, ready_klein


async def make_creation(c: TestClient, parts, **kw) -> dict:
    vid = ready_klein(c)
    job = c.post("/v1/generations", json=body(vid, **kw)).json()
    await processor(parts).process(job["id"])
    return c.get(f"/v1/creations/{c.get(f'/v1/jobs/{job["id"]}').json()['creation_id']}").json()


async def test_remix_settings_records_the_parent(gen_client: TestClient, parts) -> None:
    original = await make_creation(gen_client, parts, seed=11, size="wide")
    vid = original["models"][0]["version_id"]
    job = gen_client.post(
        "/v1/generations",
        json=body(
            vid, prompt="a red fox, but at night", seed=11, size="wide", parent_id=original["id"]
        ),
    ).json()
    await processor(parts).process(job["id"])
    done = gen_client.get(f"/v1/jobs/{job['id']}").json()
    child = gen_client.get(f"/v1/creations/{done['creation_id']}").json()
    assert child["parent_id"] == original["id"]
    assert child["params"]["seed"] == 11 and child["params"]["mode"] == "text"
    assert original["parent_id"] is None


async def test_edit_uses_the_creations_own_picture(gen_client: TestClient, parts) -> None:
    original = await make_creation(gen_client, parts, size="portrait")
    seen: dict = {}

    class Recorder(StubRunner):
        async def run(self, request, job_id, on_progress, inputs=None):
            seen["request"], seen["inputs"] = request, inputs
            return await super().run(request, job_id, on_progress, inputs)

    vid = original["models"][0]["version_id"]
    job = gen_client.post(
        "/v1/generations",
        json={
            "provider_id": "comfyui-local", "version_id": vid, "prompt": "make it dusk",
            "mode": "edit", "source_creation_id": original["id"],
        },
    ).json()  # fmt: skip
    await processor(parts, Recorder()).process(job["id"])

    assert seen["request"]["workflow"] == "flux2-klein-4b-edit"
    assert seen["request"]["source_key"] == f"creations/{original['id']}/image.png"
    source = gen_client.get(original["image_url"]).content
    assert seen["inputs"]["image"] == source  # exactly the stored picture
    done = gen_client.get(f"/v1/jobs/{job['id']}").json()
    child = gen_client.get(f"/v1/creations/{done['creation_id']}").json()
    assert child["parent_id"] == original["id"] and child["params"]["mode"] == "edit"
    assert Image.open(io.BytesIO(gen_client.get(child["image_url"]).content)).size == (
        original["width"], original["height"],
    )  # fmt: skip


async def test_remix_sources_must_belong_to_the_caller(gen_client: TestClient, parts) -> None:
    original = await make_creation(gen_client, parts)
    vid = original["models"][0]["version_id"]
    app = gen_client.app
    app.state.settings = app.state.settings.model_copy(update={"stub_user_id": "other"})  # type: ignore[attr-defined]
    # the other user cannot even see the version, so refusal is a plain not found either way
    for extra in (
        {"mode": "edit", "source_creation_id": original["id"]},
        {"parent_id": original["id"]},
    ):
        r = gen_client.post("/v1/generations", json=body(vid, **extra))
        assert r.status_code == 404


def test_edit_validation(gen_client: TestClient) -> None:
    vid = ready_klein(gen_client)
    r = gen_client.post("/v1/generations", json=body(vid, mode="edit"))
    assert r.status_code == 422  # edit needs a source
    r = gen_client.post("/v1/generations", json=body(vid, source_creation_id="x"))
    assert r.status_code == 422  # a source is only for edit
    r = gen_client.post(
        "/v1/generations", json=body(vid, mode="edit", source_creation_id="not-a-creation")
    )
    assert r.status_code == 404


async def test_missing_source_file_fails_the_job_plainly(gen_client: TestClient, parts) -> None:
    original = await make_creation(gen_client, parts)
    vid = original["models"][0]["version_id"]
    job = gen_client.post(
        "/v1/generations",
        json=body(vid, mode="edit", source_creation_id=original["id"]),
    ).json()
    parts["storage"].objects.clear()  # the file vanished after the job was queued
    await processor(parts).process(job["id"])
    got = gen_client.get(f"/v1/jobs/{job['id']}").json()
    assert got["status"] == "failed" and "source picture" in got["error"]
