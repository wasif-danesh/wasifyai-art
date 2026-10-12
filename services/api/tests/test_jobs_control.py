import asyncio

from fastapi.testclient import TestClient
from wasifyai_art_api.comfy import ComfyError
from wasifyai_art_api.runners import StubRunner

from tests.support import body, processor, ready_klein


async def test_a_queued_job_is_canceled_at_once_and_never_runs(
    gen_client: TestClient, parts
) -> None:
    vid = ready_klein(gen_client)
    job = gen_client.post("/v1/generations", json=body(vid)).json()
    r = gen_client.post(f"/v1/jobs/{job['id']}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "canceled"
    await processor(parts).process(job["id"])  # a redelivered message does nothing
    got = gen_client.get(f"/v1/jobs/{job['id']}").json()
    assert got["status"] == "canceled" and got["creation_id"] is None
    assert parts["jobs"].creations == []
    assert gen_client.post(f"/v1/jobs/{job['id']}/cancel").status_code == 409


async def test_a_running_job_is_interrupted(gen_client: TestClient, parts) -> None:
    vid = ready_klein(gen_client)
    job = gen_client.post("/v1/generations", json=body(vid)).json()
    task = asyncio.create_task(processor(parts, StubRunner(step_s=0.4)).process(job["id"]))
    for _ in range(50):
        if parts["jobs"].jobs[job["id"]][1].status == "running":
            break
        await asyncio.sleep(0.05)
    r = gen_client.post(f"/v1/jobs/{job['id']}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "running"  # asked, not yet stopped
    await asyncio.wait_for(task, 10)
    got = gen_client.get(f"/v1/jobs/{job['id']}").json()
    assert got["status"] == "canceled" and got["creation_id"] is None
    assert parts["jobs"].creations == []
    types = [e.type for e in parts["bus"].events[job["id"]]]
    assert types[-1] == "canceled"


async def test_failed_and_canceled_jobs_can_be_retried(gen_client: TestClient, parts) -> None:
    class Boom:
        async def run(self, request, job_id, on_progress, inputs=None):
            raise ComfyError("execution_failed", "The generation failed in ComfyUI.")

        async def cancel(self, job_id: str) -> None: ...

    vid = ready_klein(gen_client)
    job = gen_client.post("/v1/generations", json=body(vid, seed=5)).json()
    assert gen_client.post(f"/v1/jobs/{job['id']}/retry").status_code == 409  # not finished
    await processor(parts, Boom()).process(job["id"])
    r = gen_client.post(f"/v1/jobs/{job['id']}/retry")
    assert r.status_code == 202
    again = r.json()
    assert again["id"] != job["id"] and again["status"] == "queued"
    assert parts["jobs"].jobs[again["id"]][1].request["seed"] == 5  # same resolved request
    await processor(parts).process(again["id"])
    assert gen_client.get(f"/v1/jobs/{again['id']}").json()["status"] == "succeeded"


async def test_job_list_is_scoped_and_newest_first(gen_client: TestClient, parts) -> None:
    vid = ready_klein(gen_client)
    first = gen_client.post("/v1/generations", json=body(vid, prompt="first")).json()
    second = gen_client.post("/v1/generations", json=body(vid, prompt="second")).json()
    listed = gen_client.get("/v1/jobs").json()["jobs"]
    assert [j["prompt"] for j in listed] == ["second", "first"]
    assert listed[0]["model_name"] == "flux-2-klein-4b" and listed[0]["mode"] == "text"
    app = gen_client.app
    app.state.settings = app.state.settings.model_copy(update={"stub_user_id": "other"})  # type: ignore[attr-defined]
    assert gen_client.get("/v1/jobs").json()["jobs"] == []
    assert gen_client.post(f"/v1/jobs/{first['id']}/cancel").status_code == 404
    assert gen_client.post(f"/v1/jobs/{second['id']}/retry").status_code == 404
