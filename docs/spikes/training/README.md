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
