import io
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from wasifyai_art_api.limits import InMemoryLimiter
from wasifyai_art_api.runners import StubRunner
from wasifyai_art_api.uploads import UploadRefused, sanitize, sweep_uploads

from tests.support import body, processor, ready_klein


def picture(fmt: str = "PNG", size: tuple[int, int] = (64, 48), exif: bool = False) -> bytes:
    im = Image.new("RGB", size, (200, 30, 30))
    out = io.BytesIO()
    if exif:
        e = Image.Exif()
        e[0x0112] = 6  # rotate 90 degrees clockwise on display
        e[0x010F] = "SecretCameraMaker"
        im.save(out, fmt, exif=e)
    else:
        im.save(out, fmt)
    return out.getvalue()


def test_sanitize_rotates_strips_metadata_and_scales() -> None:
    png, w, h = sanitize(picture("JPEG", (400, 200), exif=True), max_side=100)
    assert (w, h) == (50, 100)  # rotated to portrait, then longest side scaled to 100
    with Image.open(io.BytesIO(png)) as im:
        assert im.format == "PNG" and not im.getexif() and "exif" not in im.info
    assert b"SecretCameraMaker" not in png


@pytest.mark.parametrize(
    ("data", "status"),
    [
        (picture("GIF"), 415),
        (picture("BMP"), 415),
        (b"<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>", 422),
        (b"not an image at all", 422),
        (picture("PNG")[:40], 422),  # truncated
    ],
)
def test_sanitize_refuses(data: bytes, status: int) -> None:
    with pytest.raises(UploadRefused) as e:
        sanitize(data)
    assert e.value.status == status


def test_sanitize_refuses_decompression_bombs() -> None:
    big = Image.new("L", (6000, 5000))  # 30 megapixels, tiny when compressed
    buf = io.BytesIO()
    big.save(buf, "PNG")
    assert len(buf.getvalue()) < 200_000
    with pytest.raises(UploadRefused, match="too large"):
        sanitize(buf.getvalue())


def up(c: TestClient, data: bytes, ctype: str = "image/png"):
    return c.post("/v1/uploads/images", content=data, headers={"content-type": ctype})


def test_upload_route_stores_a_clean_picture(gen_client: TestClient, parts) -> None:
    r = up(gen_client, picture("JPEG", (3000, 1500), exif=True), "image/jpeg")
    assert r.status_code == 201
    b = r.json()
    assert (b["width"], b["height"]) == (
        1024,
        2048,
    )  # EXIF-rotated, scaled to 2048 on the long side
    stored = next(iter(parts["storage"].objects.values()))
    assert stored[:8] == b"\x89PNG\r\n\x1a\n"


def test_upload_route_limits(gen_client: TestClient, parts) -> None:
    assert up(gen_client, b"x", "text/plain").status_code == 415
    assert up(gen_client, picture("GIF"), "image/gif").status_code == 415
    assert up(gen_client, b"junk", "image/png").status_code == 422
    big = gen_client.app.state.settings.model_copy(update={"upload_max_bytes": 1000})  # type: ignore[attr-defined]
    gen_client.app.state.settings = big  # type: ignore[attr-defined]
    assert up(gen_client, b"0" * 2000).status_code == 413
    gen_client.app.state.settings = big.model_copy(update={"upload_max_bytes": 10_000_000})  # type: ignore[attr-defined]
    gen_client.app.state.limiter = InMemoryLimiter()  # type: ignore[attr-defined]
    for _ in range(60):
        assert up(gen_client, picture()).status_code == 201
    assert up(gen_client, picture()).status_code == 429


async def test_edit_from_an_upload_and_the_picture_is_deleted_after(
    gen_client: TestClient, parts
) -> None:
    vid = ready_klein(gen_client)
    upload = up(gen_client, picture("PNG", (96, 64))).json()
    seen: dict = {}

    class Recorder(StubRunner):
        async def run(self, request, job_id, on_progress, inputs=None):
            seen["inputs"] = inputs
            return await super().run(request, job_id, on_progress, inputs)

    job = gen_client.post(
        "/v1/generations",
        json=body(vid, prompt="make it blue", mode="edit", source_upload_id=upload["upload_id"]),
    ).json()
    assert len(parts["storage"].objects) == 1  # just the upload so far
    await processor(parts, Recorder()).process(job["id"])

    assert seen["inputs"]["image"][:8] == b"\x89PNG\r\n\x1a\n"
    done = gen_client.get(f"/v1/jobs/{job['id']}").json()
    c = gen_client.get(f"/v1/creations/{done['creation_id']}").json()
    assert (c["width"], c["height"]) == (96, 64)
    assert c["parent_id"] is None and c["params"]["source"] == "upload"
    # the uploaded picture is gone: files and record
    assert not any("/uploads/" in k for k in parts["storage"].objects)
    assert gen_client.delete(f"/v1/uploads/{upload['upload_id']}").status_code == 404
    again = gen_client.post(
        "/v1/generations",
        json=body(vid, mode="edit", source_upload_id=upload["upload_id"]),
    )
    assert again.status_code == 404


async def test_uploaded_picture_is_deleted_when_the_job_fails(
    gen_client: TestClient, parts
) -> None:
    from wasifyai_art_api.comfy import ComfyError  # noqa: PLC0415

    class Boom:
        async def run(self, request, job_id, on_progress, inputs=None):
            raise ComfyError("execution_failed", "failed")

        async def cancel(self, job_id: str) -> None: ...

    vid = ready_klein(gen_client)
    upload = up(gen_client, picture()).json()
    job = gen_client.post(
        "/v1/generations", json=body(vid, mode="edit", source_upload_id=upload["upload_id"])
    ).json()
    await processor(parts, Boom()).process(job["id"])
    assert gen_client.get(f"/v1/jobs/{job['id']}").json()["status"] == "failed"
    assert not any("/uploads/" in k for k in parts["storage"].objects)


def test_uploads_are_private_and_validated(gen_client: TestClient, parts) -> None:
    vid = ready_klein(gen_client)
    upload = up(gen_client, picture()).json()["upload_id"]
    r = gen_client.post(
        "/v1/generations",
        json=body(vid, mode="edit", source_upload_id=upload, source_creation_id="x"),
    )
    assert r.status_code == 422  # exactly one source
    app = gen_client.app
    app.state.settings = app.state.settings.model_copy(update={"stub_user_id": "other"})  # type: ignore[attr-defined]
    assert gen_client.delete(f"/v1/uploads/{upload}").status_code == 404
    r = gen_client.post("/v1/generations", json=body(vid, mode="edit", source_upload_id=upload))
    assert r.status_code == 404


async def test_sweeper_removes_only_old_unused_uploads(gen_client: TestClient, parts) -> None:
    fresh = up(gen_client, picture()).json()["upload_id"]
    old = up(gen_client, picture()).json()["upload_id"]
    parts["uploads"].rows[old].created_at = datetime.now(UTC) - timedelta(hours=30)
    removed = await sweep_uploads(parts["uploads"], parts["storage"])
    assert removed == 1
    assert old not in parts["uploads"].rows and fresh in parts["uploads"].rows
    assert len(parts["storage"].objects) == 1
