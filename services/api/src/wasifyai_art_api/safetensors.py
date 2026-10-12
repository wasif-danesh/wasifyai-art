"""Read the header of a `.safetensors` file without loading any tensor.

The format is: 8 bytes (little-endian u64 N), N bytes of JSON, then tensor data. Only the JSON is
read, and only after N is checked against a cap, so a hostile file cannot make the reader allocate
gigabytes. Pickle formats (.ckpt, .pt, .bin) can run code when loaded and are never opened."""

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

MAX_HEADER_BYTES = 16 * 1024 * 1024
KEEP_META_KEYS = (
    "modelspec.",
    "ss_base_model_version",
    "ss_output_name",
    "ss_network_",
    "ss_sd_model_name",
)
MAX_META_VALUE = 1000


class InvalidSafetensors(ValueError):
    pass


@dataclass(frozen=True)
class Header:
    tensor_count: int
    metadata: dict[str, str]  # the whitelisted, truncated part of __metadata__
    trigger_candidates: list[str]
    header_sha256: str
    size_bytes: int


def read_header(path: Path) -> Header:
    size = path.stat().st_size
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise InvalidSafetensors("file is too short to be a safetensors file")
        (n,) = struct.unpack("<Q", raw_len)
        if n == 0 or n > MAX_HEADER_BYTES:
            raise InvalidSafetensors(f"header length {n} is not acceptable")
        if 8 + n > size:
            raise InvalidSafetensors("header is longer than the file")
        raw = f.read(n)
    try:
        doc = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise InvalidSafetensors("header is not valid JSON") from e
    if not isinstance(doc, dict):
        raise InvalidSafetensors("header is not a JSON object")

    meta_raw = doc.pop("__metadata__", {})
    if not isinstance(meta_raw, dict):
        meta_raw = {}
    data_end = 0
    for name, info in doc.items():
        if not isinstance(info, dict) or "dtype" not in info or "data_offsets" not in info:
            raise InvalidSafetensors(f"tensor {name!r} has no dtype or offsets")
        offs = info["data_offsets"]
        if not (
            isinstance(offs, list) and len(offs) == 2 and all(isinstance(o, int) for o in offs)
        ):
            raise InvalidSafetensors(f"tensor {name!r} has invalid offsets")
        data_end = max(data_end, offs[1])
    if 8 + n + data_end > size:
        raise InvalidSafetensors("file is truncated: tensor data ends past the end of the file")

    meta = {
        k: v[:MAX_META_VALUE]
        for k, v in meta_raw.items()
        if isinstance(k, str) and isinstance(v, str) and k.startswith(KEEP_META_KEYS)
    }
    return Header(
        tensor_count=len(doc),
        metadata=meta,
        trigger_candidates=_trigger_candidates(meta_raw),
        header_sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=size,
    )


def _trigger_candidates(meta: dict) -> list[str]:
    """Words the file itself suggests as trigger words: an explicit phrase, else the most
    frequent tags of the training set (a guess; the owner confirms)."""
    phrase = meta.get("modelspec.trigger_phrase")
    if isinstance(phrase, str) and phrase.strip():
        return [phrase.strip()[:120]]
    freq = meta.get("ss_tag_frequency")
    if not isinstance(freq, str):
        return []
    try:
        data = json.loads(freq)
    except json.JSONDecodeError:
        return []
    totals: dict[str, int] = {}
    if isinstance(data, dict):
        for tags in data.values():
            if isinstance(tags, dict):
                for tag, count in tags.items():
                    if isinstance(tag, str) and isinstance(count, int):
                        totals[tag.strip()] = totals.get(tag.strip(), 0) + count
    top = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    return [t for t, _ in top if t]
