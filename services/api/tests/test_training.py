import asyncio
import io
import json
import struct

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from wasifyai_art_api.comfy import ComfyError
from wasifyai_art_api.lora_convert import NotSafetensors, split, to_comfyui, write_tensors
from wasifyai_art_api.trainer import (
    StubTrainer,
    TrainImage,
    build_config,
    caption_with_trigger,
    parse_progress,
)

from tests.support import processor

TRIGGER = "pipbot"


def png(color=(200, 30, 30), size=(64, 48)) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, "PNG")
    return b.getvalue()


def make_dataset(c: TestClient, n: int = 6, name: str = "Pip") -> dict:
    d = c.post("/v1/datasets", json={"name": name, "trigger_word": TRIGGER})
    assert d.status_code == 201
    ds = d.json()
    for i in range(n):
        r = c.post(
            f"/v1/datasets/{ds['id']}/images", content=png((i * 30 % 255, 50, 90)),
            headers={"content-type": "image/png"},
        )  # fmt: skip
        assert r.status_code == 201, r.text
    return c.get(f"/v1/datasets/{ds['id']}").json()


# ---- converter and trainer helpers --------------------------------------------------------------


def adapter(names: list[str]) -> bytes:
    return write_tensors(
        {n: ("F32", [2, 2], struct.pack("<4f", 1, 2, 3, 4)) for n in names},
        {"mflux_version": "x"},
    )


def test_converter_renames_only_the_double_block_output_projection() -> None:
    names = [
        "transformer.transformer_blocks.0.attn.to_out.lora_A.weight",
        "transformer.transformer_blocks.3.attn.to_out.lora_B.weight",
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer.single_transformer_blocks.0.attn.to_out.lora_A.weight",
    ]
    out = to_comfyui(adapter(names), title="Pip", trigger_word=TRIGGER)
    header, payload = split(out)
    meta = header.pop("__metadata__")
    assert sorted(header) == sorted(
        [
            "transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight",
            "transformer.transformer_blocks.3.attn.to_out.0.lora_B.weight",
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
            "transformer.single_transformer_blocks.0.attn.to_out.lora_A.weight",  # unchanged
        ]
    )
    assert meta["modelspec.trigger_phrase"] == TRIGGER and meta["modelspec.title"] == "Pip"
    assert meta["modelspec.architecture"] == "flux-2-klein/lora" and meta["mflux_version"] == "x"
    assert len(payload) == 4 * 16  # tensor bytes untouched
    assert struct.unpack("<Q", out[:8])[0] % 8 == 0  # aligned header


def test_converter_refuses_non_safetensors() -> None:
    with pytest.raises(NotSafetensors):
        to_comfyui(b"not a file", title="x", trigger_word="y")
    with pytest.raises(NotSafetensors):
        to_comfyui(struct.pack("<Q", 10**9) + b"{}", title="x", trigger_word="y")


def test_captions_always_carry_the_trigger_word() -> None:
    assert caption_with_trigger("a robot waving", "pipbot") == "pipbot, a robot waving"
    assert caption_with_trigger("PipBot waving", "pipbot") == "PipBot waving"
    assert caption_with_trigger("  ", "pipbot") == "pipbot"


def test_progress_is_read_only_from_the_training_bar() -> None:
    assert parse_progress("Fetching 16 files:  75%|███| 12/16 [02:11<02:12]", 300) is None
    assert parse_progress("  5%|▌| 15/300 [01:46<33:58,  7.16s/it]", 300) == 15
    assert parse_progress("5/300 [00:30<..] \r 6/300 [00:37<..]", 300) == 6


def test_config_matches_the_measured_klein_layout() -> None:
    req = {
        "model": "flux2-klein-base-4b", "epochs": 20, "save_every": 100, "steps": 300,
        "params": {"quantize": 8, "gradient_checkpointing": True, "max_resolution": 768, "rank": 16,
                   "learning_rate": 1e-4, "blocks": {"double": 5, "single": 20}},
    }  # fmt: skip
    from pathlib import Path  # noqa: PLC0415

    cfg = build_config(req, Path("/d"), Path("/o"))
    ends = {
        t["module_path"].split(".")[0]: t["blocks"]["end"] for t in cfg["lora_layers"]["targets"]
    }
    assert ends == {"transformer_blocks": 5, "single_transformer_blocks": 20}
    assert len(cfg["lora_layers"]["targets"]) == 14
    assert cfg["training_loop"]["num_epochs"] == 20 and cfg["checkpoint"]["save_frequency"] == 100
    # Subject and composition are decided at high noise, so the full range is trained.
    loop = cfg["training_loop"]
    assert (loop["timestep_low"], loop["timestep_high"], loop["timestep_type"]) == (
        0,
        40,
        "sigmoid",
    )


# ---- datasets -----------------------------------------------------------------------------------


def test_dataset_pictures_are_sanitized_and_start_with_the_trigger_caption(
    gen_client: TestClient,
) -> None:
    ds = make_dataset(gen_client, 2)
    assert ds["image_count"] == 2 and ds["trigger_word"] == TRIGGER
    img = ds["images"][0]
    assert img["caption"] == TRIGGER and img["has_trigger"] is True
    assert gen_client.get(img["thumb_url"]).headers["content-type"] == "image/jpeg"


def test_dataset_validation_and_limits(gen_client: TestClient) -> None:
    bad = gen_client.post("/v1/datasets", json={"name": "x", "trigger_word": "has space"})
    assert bad.status_code == 422
    ds = gen_client.post("/v1/datasets", json={"name": "x", "trigger_word": TRIGGER}).json()
    url = f"/v1/datasets/{ds['id']}/images"
    assert (
        gen_client.post(url, content=b"junk", headers={"content-type": "image/png"}).status_code
        == 422
    )
    assert (
        gen_client.post(url, content=b"x", headers={"content-type": "text/plain"}).status_code
        == 415
    )


def test_captions_can_be_edited_and_pictures_removed(gen_client: TestClient, parts) -> None:
    ds = make_dataset(gen_client, 2)
    iid = ds["images"][0]["id"]
    r = gen_client.patch(
        f"/v1/datasets/{ds['id']}/images/{iid}", json={"caption": "a robot waving"}
    )
    assert r.json()["caption"] == "a robot waving" and r.json()["has_trigger"] is False
    assert gen_client.delete(f"/v1/datasets/{ds['id']}/images/{iid}").status_code == 204
    assert gen_client.get(f"/v1/datasets/{ds['id']}").json()["image_count"] == 1
    assert gen_client.delete(f"/v1/datasets/{ds['id']}").status_code == 204
    assert not any("/datasets/" in k for k in parts["storage"].objects)


def test_datasets_are_private(gen_client: TestClient) -> None:
    ds = make_dataset(gen_client, 1)
    app = gen_client.app
    app.state.settings = app.state.settings.model_copy(update={"stub_user_id": "other"})  # type: ignore[attr-defined]
    assert gen_client.get(f"/v1/datasets/{ds['id']}").status_code == 404
    assert gen_client.get("/v1/datasets").json()["datasets"] == []
    assert gen_client.delete(f"/v1/datasets/{ds['id']}").status_code == 404
    r = gen_client.post(
        f"/v1/datasets/{ds['id']}/images", content=png(), headers={"content-type": "image/png"}
    )
    assert r.status_code == 404


# ---- training runs ------------------------------------------------------------------------------


def start(c: TestClient, ds: dict, **kw):
    return c.post("/v1/trainings", json={"dataset_id": ds["id"], **kw})


def test_training_needs_enough_pictures(gen_client: TestClient) -> None:
    ds = make_dataset(gen_client, 3)
    r = start(gen_client, ds)
    assert r.status_code == 409 and "at least 5" in r.json()["detail"]


def test_training_request_is_resolved_from_the_preset(gen_client: TestClient, parts) -> None:
    ds = make_dataset(gen_client, 6)
    r = start(gen_client, ds, preset="fast")
    assert r.status_code == 202
    t = r.json()
    assert (t["steps"], t["image_count"], t["preset"]) == (6 * 20, 6, "fast")
    job = parts["jobs"].jobs[t["id"]][1]
    assert job.request["kind"] == "train" and job.request["model"] == "flux2-klein-base-4b"
    assert job.request["save_every"] >= 25 and parts["bus"].queued == [t["id"]]
    listed = gen_client.get("/v1/jobs").json()["jobs"][0]
    assert listed["mode"] == "train" and listed["prompt"] == "Train LoRA: Pip"
    assert start(gen_client, ds, epochs=3).json()["steps"] == 18  # explicit epochs win


async def run_job(parts, job_id, trainer=None):
    await processor(parts, trainer=trainer or StubTrainer()).process(job_id)


async def test_stub_run_records_checkpoints_with_previews(gen_client: TestClient, parts) -> None:
    ds = make_dataset(gen_client, 5)
    t = start(gen_client, ds, epochs=10).json()  # 50 steps, a checkpoint every 25
    await run_job(parts, t["id"])
    done = gen_client.get(f"/v1/trainings/{t['id']}").json()
    assert done["status"] == "succeeded" and done["progress"] == 1.0
    assert [c["step"] for c in done["checkpoints"]] == [25, 50]
    assert all(c["preview_url"] for c in done["checkpoints"])
    assert (
        gen_client.get(done["checkpoints"][0]["preview_url"]).headers["content-type"]
        == "image/jpeg"
    )
    assert (
        gen_client.get(f"/v1/trainings?dataset_id={ds['id']}").json()["trainings"][0]["id"]
        == t["id"]
    )
    types = [e.type for e in parts["bus"].events[t["id"]]]
    assert types[0] == "queued" and types[-1] == "succeeded"


async def test_trainer_receives_captions_with_the_trigger_word(
    gen_client: TestClient, parts
) -> None:
    ds = make_dataset(gen_client, 5)
    gen_client.patch(
        f"/v1/datasets/{ds['id']}/images/{ds['images'][0]['id']}", json={"caption": "a small robot"}
    )
    seen: list[TrainImage] = []

    class Rec(StubTrainer):
        async def train(self, request, job_id, images, on_progress):
            seen.extend(images)
            return await super().train(request, job_id, images, on_progress)

    t = start(gen_client, ds, epochs=5).json()
    await run_job(parts, t["id"], Rec())
    assert len(seen) == 5 and all(TRIGGER in i.caption.lower() for i in seen)
    assert seen[0].caption == "pipbot, a small robot"
    assert seen[0].png[:8] == b"\x89PNG\r\n\x1a\n"


async def test_a_running_training_can_be_canceled(gen_client: TestClient, parts) -> None:
    ds = make_dataset(gen_client, 5)
    t = start(gen_client, ds, epochs=40).json()  # 200 steps at 20 ms
    task = asyncio.create_task(run_job(parts, t["id"], StubTrainer(step_s=0.05)))
    for _ in range(100):
        if parts["jobs"].jobs[t["id"]][1].status == "running":
            break
        await asyncio.sleep(0.05)
    assert gen_client.post(f"/v1/jobs/{t['id']}/cancel").status_code == 200
    await asyncio.wait_for(task, 15)
    got = gen_client.get(f"/v1/trainings/{t['id']}").json()
    assert got["status"] == "canceled" and got["checkpoints"] == []


async def test_trainer_failure_is_reported_plainly(gen_client: TestClient, parts) -> None:
    class Boom(StubTrainer):
        async def train(self, request, job_id, images, on_progress):
            raise ComfyError("training_failed", "The training failed.")

    ds = make_dataset(gen_client, 5)
    t = start(gen_client, ds).json()
    await run_job(parts, t["id"], Boom())
    got = gen_client.get(f"/v1/trainings/{t['id']}").json()
    assert got["status"] == "failed" and got["error"] == "The training failed."


# ---- publishing ---------------------------------------------------------------------------------


async def trained(gen_client: TestClient, parts) -> dict:
    ds = make_dataset(gen_client, 5)
    t = start(gen_client, ds, epochs=10).json()
    await run_job(parts, t["id"])
    return gen_client.get(f"/v1/trainings/{t['id']}").json()


async def test_publish_converts_places_and_registers_a_ready_lora(
    gen_client: TestClient, parts
) -> None:
    t = await trained(gen_client, parts)
    r = gen_client.post(
        f"/v1/trainings/{t['id']}/publish", json={"step": 50, "name": "Pipbot Style"}
    )
    assert r.status_code == 201, r.text
    v = r.json()
    assert (v["family"], v["status"], v["trigger_words"]) == ("flux2-klein", "ready", [TRIGGER])
    rel = v["files"][0]["rel_path"]
    assert rel.startswith("loras/pipbot-style-step50")
    on_disk = parts["settings"].models_dir / rel
    header, _ = split(on_disk.read_bytes())
    names = [k for k in header if k != "__metadata__"]
    assert any(k.endswith("attn.to_out.0.lora_A.weight") for k in names)  # converted
    assert header["__metadata__"]["modelspec.trigger_phrase"] == TRIGGER
    models = gen_client.get("/v1/models").json()["models"]
    assert any(m["name"] == "Pipbot Style" and m["kind"] == "lora" for m in models)
    # a second publish does not overwrite the first
    r2 = gen_client.post(
        f"/v1/trainings/{t['id']}/publish", json={"step": 50, "name": "Pipbot Style"}
    )
    assert r2.status_code == 201
    assert r2.json()["files"][0]["rel_path"] != rel


async def test_publish_refusals(gen_client: TestClient, parts) -> None:
    t = await trained(gen_client, parts)
    assert gen_client.post(f"/v1/trainings/{t['id']}/publish", json={"step": 7}).status_code == 404
    assert gen_client.post("/v1/trainings/nope/publish", json={"step": 50}).status_code == 404
    parts["settings"].models_dir = None
    r = gen_client.post(f"/v1/trainings/{t['id']}/publish", json={"step": 50})
    assert r.status_code == 409 and "MODELS_DIR" in r.json()["detail"]


def test_checkpoint_json_roundtrip_is_plain() -> None:
    json.dumps([{"step": 1, "adapter_key": "a", "preview_key": None, "size_bytes": 1}])


def test_rank_override_and_checkpoint_spacing(gen_client: TestClient, parts) -> None:
    ds = make_dataset(gen_client, 5)
    t = start(gen_client, ds, epochs=60, rank=64).json()  # 300 steps
    job = parts["jobs"].jobs[t["id"]][1]
    assert job.request["params"]["rank"] == 64
    assert job.request["save_every"] == 50  # about six checkpoints per run
    assert start(gen_client, ds, rank=2).status_code == 422
