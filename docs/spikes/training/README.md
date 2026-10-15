# Spike: LoRA training on the dev Mac

Question: can we train a FLUX.2 klein 4B LoRA on this Mac, and does it work on the distilled klein 4B used
for generation and in ComfyUI?

## What is already known (2026-10-12, no model downloaded)

| Finding | Source |
|---|---|
| mflux 0.22.0 installs cleanly (isolated venv) and ships `mflux-train` | installed and run (`--help`, `mflux-capabilities`) |
| It trains FLUX.2 klein **base** 4B and 9B (`flux2-klein-base-4b`); distilled models are not the training target | mflux FLUX.2 README |
| Two data layouts, auto-detected: text-to-image `NN.png` + `NN.txt`; **edit** `NN_in.png`, `NN_in.txt`, `NN_out.png` | mflux training docs. Covers the "Image model" and "Edit image" training types in `docs/DESIGN.md` |
| Resume from checkpoint zips, loss plot, periodic preview images, `model_path` for a local copy | same |
| 32-36 GB Macs need `gradient_checkpointing: true` and `quantize: 8`; this Mac has 64 GB | same |
| Weights: `black-forest-labs/FLUX.2-klein-base-4B`, Apache-2.0, **not gated**, about 15 GB (README figure for the 4B repo) | HF model card, mflux README |
| Our installed ComfyUI file is the distilled single-file klein; mflux needs the HF diffusers layout of the **base** model | file inspection |
| klein 4B = 5 double + 20 single blocks, matching mflux's example target ranges | header of the installed model |

## Results (2026-10-13, dev Mac, 64 GB, mflux 0.22.0)

| Question | Answer |
|---|---|
| Does it run on the Mac? | **Yes.** 15 images, 300 steps, rank 16, `quantize: 8`, gradient checkpointing, 768 px: **about 7-8 s/step, 36-40 min total**, no memory problems |
| Download | `black-forest-labs/FLUX.2-klein-base-4B`, about 16 GB. With the default xet backend it stalled at ~0 KB/s; with `HF_HUB_DISABLE_XET=1` it ran at ~9 MB/s (about 28 min) |
| What does a checkpoint contain? | `NNNNNNN_adapter.safetensors` (96 MB, a plain LoRA: 200 tensors, rank 16, F32, names `transformer.transformer_blocks.N.attn.to_q.lora_A.weight` ...), plus optimizer state, loss history and config. Previews and a loss plot are written next to it |
| Does the LoRA load in ComfyUI on the **distilled** klein 4B? | **Yes**, and it changes the output strongly (mean pixel diff ~21 at weight 1.0; weight 0.0 is identical to no LoRA) |
| ComfyUI key mapping? | Everything maps except the double-block `attn.to_out` (10 of 200 tensors). `convert_mflux_lora.py` renames those to `attn.to_out.0`; after that ComfyUI reports **0 unmapped keys** and the output differs (diff ~5), so those weights matter |
| Did it learn the mascot? | **Not convincingly yet.** With a long descriptive prompt the LoRA nudges the design toward the dataset (body seam, chunkier silver arms). With only the trigger word ("pipbot waving hello") the output is still a generic character, so the trigger is not bound at 300 steps |

Pictures: `results/training_previews.jpg` (base model during training, steps 0/100/200/300) and
`results/distilled_klein_with_lora.jpg` (distilled klein in ComfyUI, with and without the LoRA).

### What this means for the product

- The pipeline is real: dataset -> mflux -> `adapter.safetensors` -> converter -> ComfyUI LoRA stack.
- Quality needs tuning, not a different tool: more steps (1000+; that is ~2 hours here), higher rank or
  learning rate, and captions that always contain the trigger word. Step 200 and 300 looked alike.
- The converter belongs in the trainer service, so a published LoRA is directly usable.
- Training on the Mac blocks image generation for its whole duration (one GPU).

## Still open

1. A longer run with a tuned config: does the trigger word bind, and does it hold on the distilled model?
2. Edit (paired) training, which mflux supports with `NN_in`/`NN_out` files: not run.

## Run

```bash
python docs/spikes/training/make_dataset.py http://127.0.0.1:8191 workflows/flux2-klein-4b.json data/spikes/pipbot
docs/spikes/training/run_spike.sh        # downloads ~16 GB the first time
python docs/spikes/training/convert_mflux_lora.py <adapter.safetensors> <out.safetensors>
```

## Tuned runs through the product (2026-10-13)

Both runs went through the real feature (dataset, `balanced` preset, publish, generate on the distilled
klein in ComfyUI), same 15 pipbot pictures, seed 11, LoRA weight 1.0. Grids: `results/run2_*.jpg`, `results/run3_*.jpg`.

| Run | Captions | Steps | Trigger-only prompt ("pipbot waving hello") | Long prompt describing the robot |
|---|---|---|---|---|
| 2 | trigger + full description of the robot and the scene | 900 | generic character, as without the LoRA | subtle change (arms, seam) |
| 3 | trigger + scene only (subject traits removed) | 600 | generic character; silver arms appear at 600 | not re-tested |

- The training works: the base-model previews written during run 2 show the red robot, yellow-tipped antenna and
  silver arms by step 900.
- Raising the LoRA weight to 1.5 or 2.0 only degrades the picture; the trigger still does not bind.
- Running the run-2 adapter on the **base** model in mflux (28 steps, guidance 4) also leaves a trigger-only prompt
  almost unchanged, so transferring to the distilled model is not the main cause.
- Removing the subject description from the captions (run 3) moved things slightly, not enough.
- Not yet tried: a higher learning rate (3e-4 to 5e-4), more steps (1500+), a higher rank, a rarer trigger token,
  and more varied pictures. Dataset size (15 pictures, one rendering style) is also a likely limit.

## Trigger binding fixed: the timestep range (2026-10-14)

Runs 2-4 above did not bind the trigger. The cause was a config value copied from mflux's README example,
`timestep_low: 25, timestep_high: 40`, which trains only the low-noise end of the 40-step schedule. With klein's
sigma shift that never sees sigma above about 0.64, where composition and subject are decided, so the LoRA learned
texture and style only. Evidence: after 100 epochs the adapter could not even reproduce its own training picture
from its exact training caption.

Run 5: same 15 pictures, invented token `ZQX7K`, subject-free captions, rank 64, **full range 0-40 with
`timestep_type: sigmoid`**, 600 steps (checkpoints every 100).

| Test | Result |
|---|---|
| Base model + step 300, exact training caption | reproduces the training picture's mascot, desk and tea |
| Base model + step 300, "ZQX7K waving hello" | red robot, blue eyes, yellow-tipped antenna |
| **Distilled klein in ComfyUI**, trigger-only prompts, weight 1.0 | robot appears at step 300 and 600, also in "riding a bicycle in a park" and "ZQX7K" alone; without the LoRA the same prompts give a person or the letters "ZQX7K" |

Pictures: `results/run5_*.jpg`. Training now defaults to the full range with sigmoid sampling
(`config/training.yaml`). Run 5 differs from run 4 only in the range (and fewer steps: 600 against 1500), so the
range is the proven fix.

Run 6 (shipped defaults): `fast` preset = rank 16, 300 steps, token `pipbot` (a real-word token), same pictures and
subject-free captions. On the distilled klein in ComfyUI the trigger-only prompts ("pipbot waving hello", "pipbot
riding a bicycle in a park", "pipbot") show the red robot already at step 150 and clearly at step 300; without the
LoRA they give a human mascot, a cyclist and a jar labelled "pipbot". Picture: `results/run6_*.jpg`.
So neither a higher rank nor an invented token is needed for this dataset; the defaults work as shipped.

## Realistic person (2026-10-14, run 7)

No real person's photos were used. The dataset is a fictional woman generated with klein (one portrait, then 18
pictures of her in different scenes and outfits through the klein edit workflow; hair length and waviness drift a
little between pictures, which makes this harder than real photos). Trigger `xelmira`, captions are scene only (no
face or hair description), shipped `fast` preset: rank 16, 20 epochs = 360 steps, full timestep range. Pictures:
`results/run7_*.jpg`.

On the distilled klein in ComfyUI, weight 1.0, trigger-only prompts in scenes that are not in the dataset
("xelmira reading a newspaper in a cafe", "xelmira hiking on a mountain trail", "portrait photo of xelmira",
"xelmira"):

- Without the LoRA: a different, random woman in every picture (and the word rendered as text on the newspaper).
- With the LoRA at step 150 and at 360: the same auburn, shoulder-length-haired, freckled woman in all four scenes.
  Step 150 is already enough; step 360 is very close to it.
- Not reproduced: the small scar on the left eyebrow, and the hair is slightly less wavy than the source. Identity
  similarity was judged by eye; no face-embedding metric was run.
- Not tested: real photographs, multiple people in one picture, and weights other than 1.0.

## Photorealistic character from an external generator (2026-10-14, run 8)

Dataset: "Ena", a fictional character made with another image generator (36 pictures from the owner's
`character-lora` project: studio angles, casual wear, Indian clothing, sportswear, varied hairstyles). Trigger
`enav7qx`, captions are scene and clothing only (no face description; hairstyle named only where it varies), shipped
`fast` preset with 10 epochs = 360 steps, rank 16, full timestep range. Picture: `results/run8_*.jpg`.

Distilled klein in ComfyUI, weight 1.0, trigger-only prompts in scenes not in the dataset:

- Without the LoRA: a man (cafe, portrait), a hiker, and a green logo with the token as text.
- With the LoRA (step 150 and 360): one consistent woman in her forties or fifties with dark hair and the same
  facial structure in all four scenes. Step 150 and step 360 are close.
- Likeness is only partial against the reference pictures: the source has long, wavy, highlighted hair, while the
  generated woman has shoulder-length, straighter hair; the face is similar in type but not clearly the same
  individual. The trigger works (identity is bound and stable), the fidelity is not yet.
- Likely levers, not yet tried: more steps (36 pictures x 10 epochs shows each picture only 10 times; try 20-30
  epochs), a higher rank, naming constant traits in the prompt at generation time, weight 1.2.
