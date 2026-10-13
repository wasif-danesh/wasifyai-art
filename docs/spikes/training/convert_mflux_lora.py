"""Make an mflux-trained FLUX.2 klein LoRA fully loadable in ComfyUI.

mflux writes diffusers-style names. ComfyUI's Flux mapping recognises them except the double-block output
projection, which it expects as `attn.to_out.0` (mflux writes `attn.to_out`). This renames exactly those
keys and changes nothing else. Needs the `safetensors` package.

    python convert_mflux_lora.py <adapter.safetensors> <out.safetensors>
"""

import re
import sys

from safetensors import safe_open
from safetensors.torch import save_file

src, dst = sys.argv[1], sys.argv[2]
pattern = re.compile(r"^(transformer\.transformer_blocks\.\d+\.attn\.to_out)\.(lora_[AB]\.weight)$")
tensors, renamed = {}, 0
with safe_open(src, framework="pt") as f:
    meta = dict(f.metadata() or {})
    for key in f.keys():
        new = pattern.sub(r"\1.0.\2", key)
        renamed += new != key
        tensors[new] = f.get_tensor(key)
save_file(tensors, dst, metadata=meta)
print(f"{len(tensors)} tensors written, {renamed} renamed")
