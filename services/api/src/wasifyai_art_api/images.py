import io

from PIL import Image

THUMB_PX = 480


def reencode(data: bytes) -> tuple[bytes, int, int, bytes]:
    """The image as a metadata-free PNG, its size, and a small
    JPEG thumbnail for the gallery."""
    with Image.open(io.BytesIO(data)) as im:
        rgb = im.convert("RGB")
    width, height = rgb.size
    png = io.BytesIO()
    rgb.save(png, "PNG", optimize=True)
    thumb = rgb.copy()
    thumb.thumbnail((THUMB_PX, THUMB_PX))
    jpg = io.BytesIO()
    thumb.save(jpg, "JPEG", quality=82, optimize=True)
    return png.getvalue(), width, height, jpg.getvalue()
