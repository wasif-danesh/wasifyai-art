# wasifyai-art

Repository for **wasifyai-art**: a model hub and art generator in the style of CivitAI.
Browse and manage models and LoRAs, generate through a choice of providers, keep and remix your creations, and
train your own LoRAs. Personal-first and open source.

Part of the **wasifyai.com** family of products; this one is intended to be served at `art.wasifyai.com`.
(Deployment and DNS are not set up yet.)

It is **a separate product from wasifyai** (which implements ElevenLabs-style audio features). It uses the same software
stack and conventions so it can be ported into wasifyai later, but shares no code, database or deployment with it.

Status: first generation slice.
- The hub scans a ComfyUI models folder (`MODELS_DIR`), reads `.safetensors` headers only, registers checkpoints
  and LoRAs, and lets you confirm a base family before a model is `ready`.
- **Create** picks a provider and a ready base model, queues a job on Redis, a worker runs it on ComfyUI
  (`COMFYUI_MODE=real`) or a placeholder (`stub`), and the result appears with live progress (SSE), in the
  gallery, with the model, seed and settings it was made with.
- **Remix:** from a creation, *Remix settings* reloads its prompt, model and size (optionally the same seed;
  the same seed reproduced a pixel-identical image on the dev Mac) and records lineage; *Remix picture* runs
  image to image on the klein edit workflow using your own creation as the input.
- **LoRAs:** up to 5 per generation, spliced into the workflow by the map file's `lora_chain` (weights
  -1 to 2, same-family only, `ready` only). Remix carries them; the detail page lists them with weights.
- **Jobs:** `/jobs` lists your jobs with cancel (queued: at once; running: interrupts ComfyUI, within about a
  second) and retry. **Pictures:** upload (PNG/JPEG/WebP, 10 MB, re-encoded without metadata, 24 MP cap, 60 per
  hour) and use it for *From a picture*; the upload is deleted after the job and unused ones after 24 hours.
- **Training:** `/train` makes a dataset (pictures + captions + a trigger word), trains a klein LoRA with
  mflux on this Mac (presets fast / balanced / quality, progress, cancel), shows checkpoint previews and
  publishes a chosen step into the models folder as a `ready` LoRA that Create can use at once.
  `TRAINER_MODE=stub` simulates a run without a GPU; `real` needs the mflux install (`MFLUX_BIN`).
  Verified end to end on the dev Mac: 15 pictures, 60 steps, publish, generate with the LoRA.
- Not yet: a tuned longer run where the trigger word clearly binds, multi-picture tools, prompt
  enhance. Design: [docs/DESIGN.md](docs/DESIGN.md).

```bash
make setup     # uv + pnpm dependencies
make infra     # Postgres (pgvector) :5433, Redis :6380, SeaweedFS :8334
make migrate
make api       # http://localhost:8010   (docs at /docs)
make worker    # runs queued jobs (COMFYUI_MODE=stub|real)
make web       # http://localhost:3010
make test lint
```

| Path | Content |
|---|---|
| `apps/web` | Next.js app (BFF; the browser never calls the API directly) |
| `services/api` | FastAPI, Alembic migrations, tests |
| `packages/contracts` | Pydantic wire models (source of truth) |
| `packages/contracts-ts` | Generated TypeScript types (`make contracts`) |
| `config/providers.yaml` | Provider catalogue |
| `docs/DESIGN.md` | Product and system design |
| `docs/decisions/` | Architecture decision records for this project |
| `docs/spikes/` | Spike scripts and results (Change Clothes on klein 4B and Qwen Image 2.1) |
