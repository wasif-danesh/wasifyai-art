"""Spike dataset for LoRA training: 15 synthetic images of one made-up mascot ("pipbot"), generated with
the local ComfyUI and the klein text-to-image workflow, each with a caption that starts with the trigger
word. No real person's photo is involved. Output is the mflux layout: NN.png + NN.txt.

    python make_dataset.py <comfyui_url> <workflow.json> <out_dir>
"""

import json
import sys
import time
import urllib.parse
import urllib.request

BASE, WORKFLOW, OUT = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
TRIGGER = "pipbot"
LOOK = (
    "pipbot, a small round red robot mascot with big blue eyes, a white antenna with a yellow tip, "
    "and little silver arms"
)
SCENES = [
    "sitting on a wooden desk next to a cup of tea",
    "standing in a green meadow with daisies",
    "waving hello on a sunny city street",
    "floating in space with distant stars",
    "reading a small book in a cozy library",
    "walking on a sandy beach at sunset",
    "standing in fresh snow with falling snowflakes",
    "holding a tiny umbrella in the rain",
    "next to a bicycle in a quiet park",
    "on a kitchen counter beside a plant",
    "looking out of a window at night",
    "in a workshop surrounded by tools",
    "riding a small skateboard on a ramp",
    "sleeping on a pillow in a bedroom",
    "standing on a hill with a rainbow behind",
]


def call(path: str, data: dict | None = None) -> dict:
    req = urllib.request.Request(
        BASE + path,
        json.dumps(data).encode() if data is not None else None,
        {"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def generate(prompt: str, seed: int) -> bytes:
    g = json.load(open(WORKFLOW))
    g["4"]["inputs"]["text"] = prompt
    g["9"]["inputs"]["noise_seed"] = seed
    g["7"]["inputs"]["steps"] = 4
    for n in ("6", "7"):
        g[n]["inputs"]["width"] = g[n]["inputs"]["height"] = 1024
    pid = call("/prompt", {"prompt": g, "client_id": "dataset"})["prompt_id"]
    while True:
        h = call(f"/history/{pid}")
        if pid in h and h[pid]["status"].get("completed"):
            break
        time.sleep(1)
    f = h[pid]["outputs"]["13"]["images"][0]
    q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f["subfolder"], "type": f["type"]})
    return urllib.request.urlopen(f"{BASE}/view?{q}").read()


for i, scene in enumerate(SCENES, 1):
    prompt = f"{LOOK}, {scene}, soft natural light, clean illustration style"
    t = time.time()
    open(f"{OUT}/{i:02d}.png", "wb").write(generate(prompt, 1000 + i))
    open(f"{OUT}/{i:02d}.txt", "w").write(prompt)
    print(f"{i:02d} {time.time() - t:4.1f}s {scene}", flush=True)
open(f"{OUT}/preview_1.txt", "w").write(f"{LOOK}, waving at the camera on a plain background")
print("done")
