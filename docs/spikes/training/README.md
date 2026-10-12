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

## Open questions (need the download)

1. Does it run and converge: peak memory, seconds per step, loss falling?
2. Does a LoRA trained on the base model change the output of the **distilled** 4B at 4 steps?
3. What does a finished LoRA look like on disk (the checkpoint zip holds training state)? Is there a plain
   `.safetensors` adapter to ship to ComfyUI?
4. **ComfyUI key mapping.** mflux uses diffusers-style names (`transformer_blocks.N.attn.to_q`,
   `ff.linear_in`, `single_transformer_blocks.N.attn.to_qkv_mlp_proj`). ComfyUI 0.39.2's mapping covers
   Flux.1 names and other architectures; **no entries for these FLUX.2 names were found**. A conversion step
   to klein's native names (`double_blocks.N.img_attn.qkv`, `single_blocks.N.linear1`, fused projections) is
   probable and would live in the trainer service. Unverified until a real adapter exists.

## Run (after approving the ~15 GB download)

```bash
python docs/spikes/training/make_dataset.py http://127.0.0.1:8191 workflows/flux2-klein-4b.json data/spikes/pipbot  # done
docs/spikes/training/run_spike.sh
```

20 epochs on 15 images is 300 steps, checkpoints every 100. Measure seconds per step and peak memory, then A/B
"pipbot, ..." prompts with and without the LoRA at a fixed seed.
