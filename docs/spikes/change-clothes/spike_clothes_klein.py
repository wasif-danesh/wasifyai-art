"""ADR-0053 spike (candidate 1): Change Clothes with FLUX.2 klein 4B and two reference images.
Uses the repo's own workflows and a throwaway ComfyUI on :8191. Stdlib only."""

import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8191"
S = os.path.dirname(os.path.abspath(__file__))
WF = "/Users/wasifdanesh/Projects/wasifyai/products/wasifyai-image/workflows/"


def call(path, data=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def run(graph, label):
    t0 = time.time()
    try:
        pid = call("/prompt", {"prompt": graph, "client_id": "spike"})["prompt_id"]
    except urllib.error.HTTPError as e:
        print(label, "REJECTED", e.read().decode()[:800])
        raise
    while True:
        h = call(f"/history/{pid}")
        if (
            pid in h
            and h[pid].get("outputs") is not None
            and h[pid]["status"].get("completed", False)
            or (pid in h and h[pid]["status"].get("status_str") == "error")
        ):
            break
        time.sleep(2)
    e = h[pid]
    if e["status"].get("status_str") == "error":
        print(label, "ERROR", json.dumps(e["status"])[:1200])
        raise SystemExit(1)
    dt = time.time() - t0
    for node, out in e["outputs"].items():
        for img in out.get("images", []):
            q = urllib.parse.urlencode(
                {
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img["type"],
                }
            )
            data = urllib.request.urlopen(f"{BASE}/view?{q}").read()
            return data, dt
    raise SystemExit(f"{label}: no image")


def t2i(prompt, seed, name):
    g = json.load(open(WF + "flux2-klein-4b.json"))
    g["4"]["inputs"]["text"] = prompt
    g["9"]["inputs"]["noise_seed"] = seed
    g["7"]["inputs"]["steps"] = 4
    g["6"]["inputs"]["width"] = g["7"]["inputs"]["width"] = 768
    g["6"]["inputs"]["height"] = g["7"]["inputs"]["height"] = 1024
    data, dt = run(g, name)
    open(f"{S}/{name}.png", "wb").write(data)
    shutil.copy(f"{S}/{name}.png", f"{S}/in/{name}.png")
    print(f"{name}: {dt:.1f}s", flush=True)


def two_ref(person, garment, prompt, seed, name, steps=4):
    g = json.load(open(WF + "flux2-klein-4b-edit.json"))
    g["6"]["inputs"]["image"] = f"{person}.png"
    g["4"]["inputs"]["text"] = prompt
    g["15"]["inputs"]["noise_seed"] = seed
    g["10"]["inputs"]["steps"] = steps
    g["101"] = {"class_type": "LoadImage", "inputs": {"image": f"{garment}.png"}}
    g["102"] = {
        "class_type": "ImageScaleToTotalPixels",
        "inputs": {
            "image": ["101", 0],
            "upscale_method": "nearest-exact",
            "megapixels": 1,
            "resolution_steps": 16,
        },
    }
    g["103"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["102", 0], "vae": ["3", 0]}}
    g["104"] = {
        "class_type": "ReferenceLatent",
        "inputs": {"conditioning": ["12", 0], "latent": ["103", 0]},
    }
    g["105"] = {
        "class_type": "ReferenceLatent",
        "inputs": {"conditioning": ["13", 0], "latent": ["103", 0]},
    }
    g["16"]["inputs"]["positive"] = ["104", 0]
    g["16"]["inputs"]["negative"] = ["105", 0]
    data, dt = run(g, name)
    open(f"{S}/{name}.png", "wb").write(data)
    print(f"{name}: {dt:.1f}s  prompt={prompt!r}", flush=True)


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "inputs":
        t2i(
            "Full-body studio photograph of a young woman with short dark hair standing facing the camera, "
            "wearing a plain grey t-shirt and blue jeans, neutral relaxed pose, plain light grey background",
            11,
            "person",
        )
        t2i(
            "Product photograph of a red quilted puffer jacket laid flat on a plain white background, "
            "front view, no person, no mannequin",
            23,
            "garment",
        )
    elif step == "edit":
        prompts = {
            "edit_a": "Dress the person from image 1 in the garment shown in image 2. Keep the person's face, "
            "hair, pose, body and the background unchanged.",
            "edit_b": "The woman is wearing the red quilted puffer jacket over her grey t-shirt. "
            "Keep her face, hair, pose and the background exactly the same.",
        }
        for name, p in prompts.items():
            two_ref("person", "garment", p, 7, name)
