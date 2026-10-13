"""Runs the same scan flow against real Postgres. Needs `make infra` and `make migrate`:
WSH_INTEGRATION=1 uv run pytest services/api/tests/test_postgres_store.py
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from wasifyai_art_api.main import create_app
from wasifyai_art_api.settings import Settings

from tests.conftest import FAMILIES, FOLDERS, PROVIDERS
from tests.helpers import write_safetensors

pytestmark = pytest.mark.skipif(
    os.environ.get("WSH_INTEGRATION") != "1", reason="set WSH_INTEGRATION=1 with Postgres running"
)


def _cleanup(user: str) -> None:
    import asyncio  # noqa: PLC0415

    from sqlalchemy import text  # noqa: PLC0415
    from sqlalchemy.ext.asyncio import create_async_engine  # noqa: PLC0415

    async def go() -> None:
        engine = create_async_engine(Settings().database_url)
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM models WHERE owner_id = :o"), {"o": user})
        await engine.dispose()

    asyncio.run(go())


def test_scan_confirm_missing_restore_against_postgres(tmp_path: Path) -> None:
    user = f"it-{os.getpid()}"
    _cleanup(user)
    root = tmp_path / "models"
    lora = write_safetensors(
        root / "loras" / "pg-test-lora.safetensors", {"modelspec.architecture": "flux-2-klein/lora"}
    )
    write_safetensors(root / "checkpoints" / "pg-test-ckpt.safetensors")
    settings = Settings(
        providers_file=PROVIDERS, families_file=FAMILIES, model_folders_file=FOLDERS,
        models_dir=root, stub_user_id=user,
    )  # fmt: skip
    with TestClient(create_app(settings)) as c:
        assert c.post("/v1/models/scan").json()["added"] == 2
        assert c.post("/v1/models/scan").json()["unchanged"] == 2
        models = {m["slug"]: m for m in c.get("/v1/models").json()["models"]}
        v = models["pg-test-lora"]["versions"][0]
        assert v["family_hint"] == "flux2-klein" and v["status"] == "unconfirmed"
        r = c.patch(
            f"/v1/model-versions/{v['id']}", json={"family": "flux2-klein", "status": "ready"}
        )
        assert r.status_code == 200 and r.json()["status"] == "ready"
        saved = lora.read_bytes()
        lora.unlink()
        assert c.post("/v1/models/scan").json()["missing"] == 1
        lora.write_bytes(saved)
        assert c.post("/v1/models/scan").json()["restored"] == 1
        after = {m["slug"]: m for m in c.get("/v1/models").json()["models"]}
        assert after["pg-test-lora"]["versions"][0]["status"] == "ready"
    _cleanup(user)


async def test_creation_parent_round_trips_through_postgres() -> None:
    """Lineage (remix) is written and read back: guards the SELECT column mapping."""
    from uuid import uuid4  # noqa: PLC0415

    from sqlalchemy.ext.asyncio import create_async_engine  # noqa: PLC0415
    from wasifyai_art_api.jobs import CreationRecord, PostgresJobStore  # noqa: PLC0415

    engine = create_async_engine(Settings().database_url)
    store = PostgresJobStore(engine)
    user = f"it-parent-{os.getpid()}"
    parent, child = str(uuid4()), str(uuid4())
    try:
        for cid, pid in ((parent, None), (child, parent)):
            await store.add_creation(
                "local", user,
                CreationRecord(cid, "p", {"seed": 1}, "comfyui-local", f"{cid}/i.png", f"{cid}/t.jpg",
                               8, 8, "", parent_id=pid),
            )  # fmt: skip
        got = await store.get_creation("local", user, child)
        assert got is not None and got.parent_id == parent
        first = await store.get_creation("local", user, parent)
        assert first is not None and first.parent_id is None
    finally:
        await store.delete_creation("local", user, child)
        await store.delete_creation("local", user, parent)
        await engine.dispose()


async def test_lora_refs_round_trip_with_role_and_weight() -> None:
    from uuid import uuid4  # noqa: PLC0415

    from sqlalchemy.ext.asyncio import create_async_engine  # noqa: PLC0415
    from wasifyai_art_api.jobs import CreationRecord, PostgresJobStore  # noqa: PLC0415
    from wasifyai_art_contracts import CreationModelRef  # noqa: PLC0415

    engine = create_async_engine(Settings().database_url)
    store = PostgresJobStore(engine)
    user, cid = f"it-lora-{os.getpid()}", str(uuid4())
    refs = [
        CreationModelRef(version_id=None, name="base", family="f", rel_path="diffusion_models/b.safetensors",
                         header_sha256="h1"),
        CreationModelRef(version_id=None, name="kaia", family="f", rel_path="loras/k.safetensors",
                         header_sha256="h2", role="lora", weight=0.7),
    ]  # fmt: skip
    try:
        await store.add_creation(
            "local", user,
            CreationRecord(cid, "p", {}, "comfyui-local", f"{cid}/i.png", f"{cid}/t.jpg", 8, 8, "",
                           models=refs),
        )  # fmt: skip
        got = await store.get_creation("local", user, cid)
        assert got is not None
        assert [(m.role, m.name, m.weight) for m in got.models] == [
            ("base", "base", None), ("lora", "kaia", 0.7),
        ]  # fmt: skip
    finally:
        await store.delete_creation("local", user, cid)
        await engine.dispose()


def test_dataset_and_training_run_against_postgres(tmp_path: Path) -> None:
    user = f"it-train-{os.getpid()}"
    import asyncio  # noqa: PLC0415

    from sqlalchemy import text  # noqa: PLC0415
    from sqlalchemy.ext.asyncio import create_async_engine  # noqa: PLC0415

    async def clean() -> None:
        engine = create_async_engine(Settings().database_url)
        async with engine.begin() as conn:
            for table in ("training_runs", "datasets"):
                await conn.execute(text(f"DELETE FROM {table} WHERE owner_id = :o"), {"o": user})
            await conn.execute(text("DELETE FROM jobs WHERE owner_id = :o"), {"o": user})
        await engine.dispose()

    asyncio.run(clean())
    settings = Settings(
        providers_file=PROVIDERS, families_file=FAMILIES, model_folders_file=FOLDERS,
        models_dir=tmp_path / "models", data_dir=tmp_path / "data", stub_user_id=user,
        trainer_mode="stub",
    )  # fmt: skip
    from io import BytesIO  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    def png(shade: int) -> bytes:
        buf = BytesIO()
        Image.new("RGB", (64, 64), (shade, 60, 90)).save(buf, "PNG")
        return buf.getvalue()

    try:
        with TestClient(create_app(settings)) as c:
            d = c.post("/v1/datasets", json={"name": "pg set", "trigger_word": "pgtrig"}).json()
            for i in range(5):
                r = c.post(
                    f"/v1/datasets/{d['id']}/images", content=png(40 * i),
                    headers={"content-type": "image/png"},
                )  # fmt: skip
                assert r.status_code == 201, r.text
            got = c.get(f"/v1/datasets/{d['id']}").json()
            assert got["image_count"] == 5
            assert all(i["caption"] == "pgtrig" for i in got["images"])
            first = got["images"][0]["id"]
            r = c.patch(
                f"/v1/datasets/{d['id']}/images/{first}", json={"caption": "pgtrig on a table"}
            )
            assert r.status_code == 200
            run = c.post("/v1/trainings", json={"dataset_id": d["id"], "epochs": 2}).json()
            assert run["steps"] == 10
            assert run["created_at"]
            assert [t["id"] for t in c.get("/v1/trainings").json()["trainings"]] == [run["id"]]
            assert c.delete(f"/v1/datasets/{d['id']}").status_code == 204
    finally:
        asyncio.run(clean())
