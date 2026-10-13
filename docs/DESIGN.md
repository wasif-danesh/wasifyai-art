# wasifyai-art: model hub and art generator — design (draft 1)

Status: **proposal**, 2026-10-12. Repository `wasifyai-art`, product id `wasifyai-art`.
A separate product from wasifyai: wasifyai implements ElevenLabs-style audio features, this implements
model hub and art features. It shares no code, database or deployment with wasifyai, and uses the same software
stack and conventions so it can be ported into wasifyai later (section 9).

---

## 1. What it is

A **model hub plus art generation workspace** in the family of CivitAI: you keep a catalog of models
(checkpoints, LoRAs), generate with them through a choice of **providers**, keep and remix your creations, and train
your own LoRAs. Personal-first and open source; hub features are built so they can be opened to others later.

Features that belong to a commercial service (credits, Pro, priority queue, likes and comments) are left out.

### Core objects

| Object | Meaning |
|---|---|
| **Provider** | Where generation runs: local ComfyUI, a remote ComfyUI (Comfy Cloud and similar), an OpenAI-compatible image API, later others |
| **Model** | A named thing in the hub: `checkpoint` or `lora` (later `embedding`, `workflow`) |
| **Model version** | Files, base family, trigger words, recommended settings, preview images, licence note, tags |
| **Availability** | Whether a version can run on a given provider (file present locally, remote reference, or not available) |
| **Creation** | An image (later video) with its exact parameters, models used, provider, and lineage (remixed from) |
| **Dataset / Training run** | Images + captions, and a job that produces a LoRA version |
| **Tool** (v1.5) | A manifest-driven single-purpose workflow, first "Change Clothes" |

---

## 2. Scope by version

| | v1 | v1.5 | v2 |
|---|---|---|---|
| Home | Gallery of your creations, **Create** button, search | Collections, tags | Video cards |
| Creation detail | Models used, params, **Remix** (settings / picture) | Compare two creations | |
| Art editor | Text to image, Image to image, Edit; provider dropdown; base model; LoRA stack; prompt enhance | Upscale, hires | ControlNet, embeddings, video tab |
| Model hub | Browse, filter, detail, versions, previews, import from local folders, upload `.safetensors`, "made with this model" | Import by URL (CivitAI / Hugging Face), duplicates by hash | Public sharing, ratings, downloads by others |
| Providers | Local ComfyUI | Remote ComfyUI, OpenAI-compatible | More |
| Training | **Image LoRA** (Mac-first) | **Edit LoRA** (paired data) | Video LoRA (needs CUDA) |
| AI Tools | | Change Clothes | More tools |
| Not planned | Credits, Pro, priority queue, likes, comments, follows | | |

---

## 3. Providers: the dropdown

A provider declares what it can do; the UI shows only what the chosen provider supports.

```
Provider:  { id, kind, title, capabilities, limits, status }
capabilities: text2img, img2img, edit, video, loras, lora_stack_max, negative_prompt, seed, sizes[], samplers[]
```

- The **provider dropdown** sits at the top of the editor. Changing it re-filters the model list and the controls.
- **Model availability is per provider.** A LoRA file in the local ComfyUI folder is available locally; on a remote
  provider it needs a remote reference or upload; an OpenAI-compatible API cannot use LoRAs at all and says so.
  Unavailable models appear greyed with the reason, never silently dropped.
- Providers are config + adapters (`providers/<kind>.py`), the same shape as wasifyai's backends, so adding one does
  not touch the editor or the hub.
- **Two meanings of "provider" are supported:** *where it runs* (this section) and *which base model*. Both are
  dropdowns; base models are the checkpoint models the chosen provider offers.

---

## 4. Model hub

- **Sources:** (1) scan the local ComfyUI model folders (rescan button; found files start `unconfirmed`); (2) upload
  through the UI; (3) trained in-app; (4, v1.5) import by URL from CivitAI or Hugging Face with the user's own API
  key; the importer reads the source's metadata (base model, trigger words, previews).
- **Safety of files:** `.safetensors` only (pickle formats can run code on load); hash on arrival; header parsed
  for architecture and trigger-word hints; the family is a *suggestion* the user confirms, because a LoRA used on
  the wrong family fails silently.
- **Model page:** previews, versions with changelog, base family, trigger words, recommended settings (steps, CFG,
  sampler, size, positive/negative prompt), licence note (free text you research), tags, "Use in editor",
  "Made with this model" (your creations that used it), availability per provider.
- **Search:** one index over creations and models, multilingual embeddings, exact-word check, as in wasifyai's search
  design (kept portable: a table with `kind`, `item_id`, `text`, `embedding`).
- **Visibility** exists from day one (`private` default) so opening the hub later is a UI and policy change, not a
  schema change.

---

## 5. Art editor

Tabs: **Text to image · Image to image · Edit** (Video disabled until v2). One layout:

```
┌ Left panel ─────────────────────┐ ┌ Prompt bar ──────────────────────────────────────┐
│ Provider  [Local ComfyUI ▾]     │ │ prompt…                              ✨ Enhance   │
│ Base model [FLUX.2 klein 4B ▾]  │ │ ▸ Negative (if the model uses one)  1 image ▾     │
│ LoRAs (+) weight sliders        │ │                       [ Generate · ~45 s ]       │
│  trigger-word chips             │ └──────────────────────────────────────────────────┘
│ Size · Seed · Advanced          │ ┌ Results ─────────────────────────────────────────┐
│ Presets · Paste data · Reset    │ │ queued → progress → images; Remix · Publish · ⋯  │
└─────────────────────────────────┘ └──────────────────────────────────────────────────┘
```

- Controls come from a **model capability descriptor** (distilled models hide CFG and negative prompt).
- **Estimated time** replaces credits.
- **Prompt enhance:** one fixed LLM behind an alias (default Gemma 4 E4B through LiteLLM and Ollama); result is
  English, shown with Undo; LoRA trigger words are re-checked afterwards.
- **Remix** (from a creation): settings (prompt, provider, model, LoRAs, weights, size, seed keep/re-roll) or
  picture (use the creation as the input of Image to image).

---

## 6. LoRA training

Wizard: **Images → Captions → Settings → Train → Results/Publish.**

- **Types:** Image (v1), Edit with paired start/end images and optional instruction text (v1.5), Video (v2).
- **Dataset:** drag/drop, paste, or ZIP (size and file-count caps, no nested folders, zip-bomb and path-traversal
  checks); each image re-encoded and stripped of metadata; duplicates and small images flagged; per-image repeat.
- **Captions:** auto-generated by the platform's multimodal LLM alias, always editable, bulk find/replace, caption
  prompt editable by the admin; trigger word injected and validated. Captions are normalised to English (known
  multilingual limitation, recorded).
- **Settings:** simple view = **Repeat** and **Epoch** with the visible total (`images × repeat × epoch` = steps),
  plus presets; Professional view = rank, learning rate, resolution, batch. Preview prompt and sample every N epochs.
- **Training runs on the trainer service**, a native process on the Mac like wasifyai's MuseTalk server (Metal cannot
  be reached from containers); CUDA container later. One GPU: training blocks generation, and the UI says so.
- **Publish:** pick an epoch → file copied into the provider's LoRA folder → model version created → `ready` once
  the provider lists it.
- **Engine (spike, run 2026-10-13):** **mflux** (MLX) trains FLUX.2 klein base 4B on the dev Mac: about 7-8 s per
  step, 300 steps in under 40 minutes, no memory trouble. It writes a plain LoRA `adapter.safetensors`, which loads
  on the **distilled** klein in ComfyUI and changes the output; one key group needs a small rename
  (`convert_mflux_lora.py`, to live in the trainer). Dataset layouts cover both "Image model" and "Edit image".
  Not yet shown: that the trigger word binds (needs a longer, tuned run). Download: use `HF_HUB_DISABLE_XET=1` on
  slow links. Details and pictures: `docs/spikes/training/README.md`.
- **Built (2026-10-13):** `routes_training.py` (datasets, images, runs, checkpoints, publish), `trainer.py`
  (`MfluxTrainer` subprocess, `StubTrainer`), `lora_convert.py` (header-only rename, no torch), worker path for
  `kind=train` (GPU lock, cancel, progress from tqdm), `config/training.yaml` (family presets and defaults),
  web `/train`. Steps = images × epochs; checkpoints are saved about four times per run and appear when the
  run ends (the final step has no preview). Publish writes `{MODELS_DIR}/loras/{slug}-step{N}.safetensors`,
  scans it into the registry as family `flux2-klein` with the trigger word, and ComfyUI lists it without a
  restart. Verified on the real stack: 15 pictures × 4 epochs = 60 steps trained in about 15 minutes.

---

## 7. Architecture (wasifyai stack, own deployment)

```
Browser ── Next.js 15 / React 19 (BFF route handlers) ──► FastAPI (REST + SSE)
                                                           │   ├─ Postgres (+pgvector)
                                                           │   ├─ Redis (queue, pub/sub, event log)
                                                           │   ├─ Object storage (S3 API; SeaweedFS locally)
                                                           │   └─ LiteLLM ─► Ollama (enhance, captions, embeddings)
                                                           ▼
                                                    Worker (Redis streams)
                                                     ├─ provider adapters ─► ComfyUI (local / remote), others
                                                     └─ trainer client ─────► trainer server (native, Mac)
```

| Layer | Choice |
|---|---|
| Frontend | Next.js App Router, React 19, TypeScript strict, Tailwind + shadcn/ui, react-hook-form + Zod, TanStack Table, custom SSE hook |
| API | Python 3.12, FastAPI, Pydantic v2 (source of truth) → `openapi-typescript`, Alembic, uv |
| Jobs | Redis Streams with consumer groups, one job per GPU id, results published before ack, idempotent redelivery |
| Realtime | SSE; the event log in Redis so any API replica can serve a stream |
| LLM | Everything through LiteLLM aliases (`prompt-enhancer`, `multimodal`, `embedder`); admin-overridable |
| Media | ComfyUI API-format workflows + `*.map.yaml` input maps; no node IDs in Python |
| Storage | `{tenant}/{product}/{user}/…` keys through one interface (obstore); URLs, never bytes, in events |
| Orchestration | Plain service functions behind a `Run` interface. LangGraph is **not** used in v1; a graph wrapper is a thin adaptation if ported |
| Dev/ops | Podman + compose, pnpm + Turborepo, ruff/pyright/biome/vitest, Helm later |

Repository layout: `apps/web`, `services/api` (the API app and, for now, the worker: `python -m
wasifyai_art_api.worker`, a separate process that can be split out), `services/trainer` (later), `packages/contracts`,
`config/` (providers, families, model folders), `workflows/`, `docs/`.

---

## 8. Data model (first cut)

```
providers(id, kind, title, config jsonb, status)
models(id, owner, kind, name, slug, tags[], visibility, created_at)
model_versions(id, model_id, label, family, sha256, size, trigger_words[], recommended jsonb,
               licence_note, preview_keys[], status[unconfirmed|ready|missing|disabled], source)
model_files(id, version_id, file_key|local_path, kind)
availability(version_id, provider_id, status, remote_ref, checked_at)
creations(id, owner, kind[image|video|tool], prompt, params jsonb, provider_id, parent_id,
          image_key, thumb_key, width, height, created_at)
creation_models(creation_id, version_id, role, weight, file_snapshot, sha256)
jobs(id, kind[generate|caption|train|scan], status, request jsonb, provider_id, gpu_id, progress, error, …)
datasets, dataset_images(caption, caption_source, repeat, phash, flags), training_runs, training_checkpoints
usage_events(kind, quantity, unit, meta)           search_index(kind, item_id, text, embedding, model)
```
Every table carries `tenant_id` and `product_id` (constant here) to keep porting trivial.

---

## 9. Staying portable to wasifyai

Cheap now, expensive later, so done from day one:

1. `tenant_id` + `product_id` on every row, object key and job; identity resolved on every request.
2. Usage events for every billable thing (LLM tokens, GPU seconds, jobs).
3. Workflows and map files in wasifyai's exact format; capabilities and variants named, never model files in code.
4. SSE event shapes compatible with wasifyai's contract; URLs not bytes.
5. Every creation kind indexed for search; multilingual by default.
6. One switch for moderation and quotas (off locally, forceable on).
7. Providers and trainer behind interfaces, so wasifyai's media worker can replace this project's worker.

Porting then means: copy the product package (routes, schemas, prompts, workflows, tables), replace this project's
worker/queue/storage with wasifyai's platform, keep the UI under wasifyai's side panel.

---

## 10. Requirements review outcomes

| Requirement | Decision |
|---|---|
| New home card "Image & Video Art" | Art is this project's home, not a wasifyai card |
| Art home = gallery + Create | Yes |
| Detail page shows base model + LoRAs, Remix | Yes, plus provider, params, lineage |
| Editor page | Yes, controls driven by provider + model capabilities |
| Online Training under Create menu | Training is a top-level section |
| Training types Image / Video / Edit | Image v1, Edit v1.5, Video v2 |
| Labelling with generated captions and edit | Yes |
| Credits, Pro, Priority queue | Dropped; estimated time and queue position |

## 11. Decisions (2026-10-12)

1. **Conventions only:** no import of wasifyai code; same stack, file formats and contracts.
2. **Provider = where it runs;** base models are a second dropdown. Vendor APIs are just more providers.
3. **Hub v1 = local registry** (scan, upload, versions, previews, "made with this model"); CivitAI/HF import in v1.5.
4. Project name and location (decided: `~/Projects/wasifyai-art`, product `wasifyai-art`).

Superseded: wasifyai ADR-0051 to 0054 (Proposed, committed in wasifyai) described the same ideas as wasifyai products.
Their content is reusable here; they should be marked Withdrawn in wasifyai if you agree.
