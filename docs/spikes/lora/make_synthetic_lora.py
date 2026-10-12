"""Write a small random LoRA with the real FLUX.2 klein 4B layer names, to test the LoRA path end to end
without downloading anything. Run with a Python that has torch and safetensors (ComfyUI's venv).

    python make_synthetic_lora.py <klein.safetensors> <out.safetensors>

Keys use ComfyUI's `diffusion_model.<layer>.lora_A/B.weight` form. The effect is noise, not a style."""

import json
import struct
import sys

import torch
from safetensors.torch import save_file

model, out = sys.argv[1], sys.argv[2]
with open(model, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
torch.manual_seed(0)
rank, tensors = 16, {}
suffixes = (".img_attn.qkv.weight", ".img_attn.proj.weight", ".txt_attn.qkv.weight", ".txt_attn.proj.weight")
for key, info in header.items():
    if key.startswith("double_blocks.") and key.endswith(suffixes):
        out_f, in_f = info["shape"]
        base = key[: -len(".weight")]
        tensors[f"diffusion_model.{base}.lora_A.weight"] = (torch.randn(rank, in_f) / in_f**0.5).half()
        tensors[f"diffusion_model.{base}.lora_B.weight"] = (torch.randn(out_f, rank) * 0.02).half()
meta = {
    "modelspec.architecture": "flux-2-klein/lora",
    "modelspec.title": "Synthetic noise LoRA",
    "modelspec.trigger_phrase": "synthx",
}
save_file(tensors, out, metadata=meta)
print(len(tensors), "tensors written")
