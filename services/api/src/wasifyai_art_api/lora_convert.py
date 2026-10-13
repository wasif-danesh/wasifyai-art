"""Make an mflux-trained FLUX.2 klein LoRA directly usable in ComfyUI, without torch.

A safetensors file is an 8-byte length, a JSON header, then raw tensor bytes; the header's
offsets are relative to the data section. So renaming tensors only means rewriting the header
and copying the bytes.

mflux writes diffusers-style names. ComfyUI's Flux mapping recognises them except the
double-block output projection, which it expects as `attn.to_out.0` (measured on ComfyUI
0.39.2: 10 of 200 tensors)."""

import json
import re
import struct
from typing import Any

MAX_HEADER = 16 * 1024 * 1024
_DOUBLE_OUT = re.compile(
    r"^(transformer\.transformer_blocks\.\d+\.attn\.to_out)\.(lora_[AB]\.weight)$"
)


class NotSafetensors(ValueError):
    pass


def split(data: bytes) -> tuple[dict[str, Any], bytes]:
    if len(data) < 8:
        raise NotSafetensors("too short")
    (n,) = struct.unpack("<Q", data[:8])
    if n == 0 or n > MAX_HEADER or 8 + n > len(data):
        raise NotSafetensors("bad header length")
    try:
        header = json.loads(data[8 : 8 + n])
    except json.JSONDecodeError as e:
        raise NotSafetensors("header is not JSON") from e
    if not isinstance(header, dict):
        raise NotSafetensors("header is not an object")
    return header, data[8 + n :]


def join(header: dict[str, Any], payload: bytes) -> bytes:
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)  # keep the data section 8-byte aligned, as other writers do
    return struct.pack("<Q", len(raw)) + raw + payload


def to_comfyui(data: bytes, *, title: str, trigger_word: str) -> bytes:
    """The adapter with ComfyUI-compatible names and the metadata the hub reads as hints."""
    header, payload = split(data)
    meta = {k: v for k, v in header.pop("__metadata__", {}).items() if isinstance(v, str)}
    out: dict[str, Any] = {}
    for name, info in header.items():
        out[_DOUBLE_OUT.sub(r"\1.0.\2", name)] = info
    meta.update(
        {
            "modelspec.architecture": "flux-2-klein/lora",
            "modelspec.title": title[:120],
            "modelspec.trigger_phrase": trigger_word,
        }
    )
    return join({"__metadata__": meta, **out}, payload)


def write_tensors(
    tensors: dict[str, tuple[str, list[int], bytes]], metadata: dict[str, str]
) -> bytes:
    """A minimal safetensors writer: name -> (dtype, shape, raw bytes)."""
    header: dict[str, Any] = {"__metadata__": metadata}
    payload = b""
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload += raw
    return join(header, payload)
