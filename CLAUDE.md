# wasifyai-art: instructions for Claude Code

A CivitAI-style model hub and art generator (product id `wasifyai-art`). **Not part of wasifyai.** Do not edit
the wasifyai repository from here. Read `docs/DESIGN.md` before non-trivial work; decisions are ADRs in `docs/decisions/`.

## Rules
1. Same stack and conventions as wasifyai, **no wasifyai code imports**: Next.js (App Router, React 19, strict TS) as BFF;
   FastAPI + Pydantic v2 (contracts are the source of truth, TS types generated); Postgres + pgvector; Redis Streams
   for jobs and the SSE event log; S3-compatible storage through one interface; LLM calls only through LiteLLM aliases.
2. Capabilities, not models: no model names, checkpoint files or ComfyUI node IDs in Python. Workflows are API-format
   JSON with a `*.map.yaml` input/output map.
3. GPU work is never awaited inline: it is a job. One job at a time per GPU id.
4. Tenancy and identity from day one: `tenant_id` and `product_id` on every row, job and storage key; identity on every
   request; usage events for LLM tokens, GPU seconds and jobs.
5. Files go through the storage interface; events carry URLs, never bytes. Only `.safetensors` model files.
6. Every kind of creation is searchable in the library and everything is multilingual by default.
7. Moderation and quotas sit behind one switch (off locally).
8. Providers (where generation runs) and the trainer are behind interfaces.
9. Secrets only from environment variables; never commit or print them.
10. Ask before adding a service, datastore or external dependency.

## Conventions
Python 3.12 + uv + ruff + pyright + pytest; pnpm + Turborepo + biome + vitest; Alembic migrations (never edit an applied
one); structured JSON logs with request/trace/tenant/product ids; Conventional Commits. The owner commits and pushes:
finish work with a `git add` line and a single-line commit message, never commit on your own.
Create directories only when there is code to put in them.
