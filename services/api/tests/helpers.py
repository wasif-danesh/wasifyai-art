"""Build real `.safetensors` files for tests: a valid header and zeroed tensor data, no ML library."""

import json
import struct
from pathlib import Path


def write_safetensors(path: Path, metadata: dict[str, str] | None = None, tensors: int = 2) -> Path:
    header: dict = {}
    offset = 0
    for i in range(tensors):
        header[f"layer.{i}.weight"] = {
            "dtype": "F16",
            "shape": [4],
            "data_offsets": [offset, offset + 8],
        }
        offset += 8
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * offset)
    return path
