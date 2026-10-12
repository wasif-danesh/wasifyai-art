from typing import Any

import pytest
from fastapi.testclient import TestClient
from wasifyai_art_api.runners import ComfyRunner
from wasifyai_art_api.workflows import WorkflowRegistry

from tests.conftest import WORKFLOWS
from tests.support import body, processor, ready_klein, ready_lora, versions


def graph(loras: list[tuple[str, float]], flow: str = "flux2-klein-4b") -> dict[str, Any]:
    wf = WorkflowRegistry.load(WORKFLOWS).get(flow)
    assert wf is not None
    return wf.fill({"model": "m.safetensors", "prompt": "p", "seed": 1, "steps": 4}, loras)


def test_no_loras_leaves_the_graph_unchanged() -> None:
    g = graph([])
    assert g["10"]["inputs"]["model"] == ["1", 0]
    assert not any(n["class_type"] == "LoraLoaderModelOnly" for n in g.values())


def test_loras_are_chained_in_order_between_loader_and_sampler() -> None:
    g = graph([("a.safetensors", 0.8), ("sub/b.safetensors", -0.5), ("c.safetensors", 1.0)])
    loaders = {k: v for k, v in g.items() if v["class_type"] == "LoraLoaderModelOnly"}
    assert len(loaders) == 3
    first, second, third = sorted(loaders, key=int)
    assert g[first]["inputs"] == {
        "model": ["1", 0], "lora_name": "a.safetensors", "strength_model": 0.8,
    }  # fmt: skip
    assert g[second]["inputs"]["model"] == [first, 0]
    assert g[second]["inputs"]["lora_name"] == "sub/b.safetensors"
    assert g[second]["inputs"]["strength_model"] == -0.5
    assert g[third]["inputs"]["model"] == [second, 0]
    assert g["10"]["inputs"]["model"] == [third, 0]  # the sampler gets the patched model
    assert int(first) > 13  # fresh ids above the existing ones


def test_edit_workflow_splices_too() -> None:
    wf = WorkflowRegistry.load(WORKFLOWS).get("flux2-klein-4b-edit")
    assert wf is not None and wf.problems() == []
    g = wf.fill({"model": "m", "image": "x [temp]"}, [("a.safetensors", 1.0)])
    lora_id = next(k for k, v in g.items() if v["class_type"] == "LoraLoaderModelOnly")
    assert g["16"]["inputs"]["model"] == [lora_id, 0]


def test_too_many_loras_and_unsupported_workflows_are_errors() -> None:
    with pytest.raises(ValueError, match="at most 5"):
        graph([(f"{i}.safetensors", 1.0) for i in range(6)])
    wf = WorkflowRegistry.load(WORKFLOWS).get("flux2-klein-4b")
    assert wf is not None
    wf.map.lora_chain = None
    with pytest.raises(ValueError, match="does not accept"):
        wf.fill({}, [("a.safetensors", 1.0)])


def test_a_chain_that_does_not_match_the_graph_is_reported() -> None:
    wf = WorkflowRegistry.load(WORKFLOWS).get("flux2-klein-4b")
    assert wf is not None
    wf.graph["10"]["inputs"]["model"] = ["2", 0]  # someone rewired the workflow
    assert any("does not take the model" in p for p in wf.problems())


async def test_lora_flow_records_models_with_weights(gen_client: TestClient, parts) -> None:  # noqa: ANN001
    base = ready_klein(gen_client)
    lora = ready_lora(gen_client, "kaia")
    r = gen_client.post(
        "/v1/generations", json=body(base, loras=[{"version_id": lora, "weight": 0.8}])
    )
    assert r.status_code == 202
    job = r.json()
    request = parts["jobs"].jobs[job["id"]][1].request
    assert request["loras"][0]["file"] == "kaia.safetensors"
    await processor(parts).process(job["id"])
    done = gen_client.get(f"/v1/jobs/{job['id']}").json()
    c = gen_client.get(f"/v1/creations/{done['creation_id']}").json()
    assert c["params"]["lora_count"] == 1
    roles = [(m["role"], m["name"], m["weight"]) for m in c["models"]]
    assert roles == [("base", "flux-2-klein-4b", None), ("lora", "kaia", 0.8)]


class FakeClient:
    def __init__(self) -> None:
        self.graph: dict | None = None

    async def run(self, graph, client_id, on_progress, timeout_s):  # noqa: ANN001
        self.graph = graph
        return {"13": {"images": [{"filename": "x.png", "subfolder": "", "type": "output"}]}}

    @staticmethod
    def pick_image(outputs, node):  # noqa: ANN001
        from wasifyai_art_api.comfy import ComfyClient  # noqa: PLC0415

        return ComfyClient.pick_image(outputs, node)

    async def fetch(self, f):  # noqa: ANN001
        return b"png"

    async def free(self) -> None: ...


async def test_comfy_runner_sends_the_spliced_graph() -> None:
    client = FakeClient()
    runner = ComfyRunner(client, WorkflowRegistry.load(WORKFLOWS), 5.0)  # type: ignore[arg-type]

    async def progress(_p: float) -> None: ...

    await runner.run(
        {
            "workflow": "flux2-klein-4b", "model_file": "m.safetensors", "prompt": "p", "seed": 1,
            "steps": 4, "width": 1024, "height": 1024,
            "loras": [{"file": "kaia.safetensors", "weight": 0.7}],
        },
        "job1",
        progress,
    )  # fmt: skip
    assert client.graph is not None
    names = [n["inputs"]["lora_name"] for n in client.graph.values() if "lora_name" in n["inputs"]]
    assert names == ["kaia.safetensors"]


@pytest.mark.parametrize(
    ("setup", "status", "text"),
    [
        ("wrong_family", 422, "cannot be used here"),
        ("unconfirmed", 409, "not ready"),
        ("checkpoint_as_lora", 422, "not a LoRA"),
        ("too_many", 422, "at most 5"),
        ("weight", 422, "weight must be"),
        ("duplicate", 422, "once"),
        ("missing", 404, "not found"),
    ],
)
def test_lora_refusals(gen_client: TestClient, setup: str, status: int, text: str) -> None:
    base = ready_klein(gen_client)
    names = ["kaia", "lora-b", "lora-c", "lora-d", "lora-e", "lora-f"]
    if setup == "wrong_family":
        loras = [{"version_id": ready_lora(gen_client, "sdxl-thing", "sdxl")}]
    elif setup == "unconfirmed":
        loras = [{"version_id": versions(gen_client)["kaia"]}]
    elif setup == "checkpoint_as_lora":
        loras = [{"version_id": base}]
    elif setup == "too_many":
        loras = [{"version_id": ready_lora(gen_client, n)} for n in names]
    elif setup == "weight":
        loras = [{"version_id": ready_lora(gen_client, "kaia"), "weight": 2.5}]
    elif setup == "duplicate":
        v = ready_lora(gen_client, "kaia")
        loras = [{"version_id": v}, {"version_id": v}]
    else:
        loras = [{"version_id": "00000000-0000-0000-0000-000000000000"}]
    r = gen_client.post("/v1/generations", json=body(base, loras=loras))
    assert r.status_code == status, r.text
    assert text in str(r.json()["detail"])
