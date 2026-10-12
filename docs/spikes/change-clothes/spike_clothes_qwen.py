"""ADR-0053 spike (candidate 2): Qwen Image 2.1 two-picture edit, graph hand-built from the bundled template."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spike_clothes_klein import S, run


def qwen(prompt, seed, steps, name):
    g = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "qwen_image_2.1_int8_convrot.safetensors",
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_8b_int8_convrot.safetensors",
                "type": "qwen_image",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"},
        },
        "4": {"class_type": "LoadImage", "inputs": {"image": "person.png"}},
        "5": {"class_type": "LoadImage", "inputs": {"image": "garment.png"}},
        "6": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {
                "clip": ["2", 0],
                "prompt": prompt,
                "negative_prompt": "",
                "resolution": 1024,
                "images.image_1": ["4", 0],
                "images.image_2": ["5", 0],
                "vae": ["3", 0],
            },
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["6", 0],
                "negative": ["6", 1],
                "latent_image": ["6", 2],
                "seed": seed,
                "steps": steps,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1,
            },
        },
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {
            "class_type": "SaveImage",
            "inputs": {"images": ["9", 0], "filename_prefix": "qwen"},
        },
    }
    data, dt = run(g, name)
    open(f"{S}/{name}.png", "wb").write(data)
    print(f"{name}: {dt:.1f}s steps={steps}", flush=True)


qwen(
    "Keep the person in <image1> unchanged and dress them in the red puffer jacket from <image2>. "
    "Keep the face, hair, pose, jeans and background exactly the same.",
    7,
    int(sys.argv[1]) if len(sys.argv) > 1 else 8,
    "qwen_a",
)
