CONTAINER_ENGINE ?= podman
export CONTAINER_ENGINE

.PHONY: setup infra down migrate api worker web test lint format contracts
setup:                       # install Python and TypeScript dependencies
	uv sync --all-packages
	pnpm install
infra:                       # Postgres (pgvector), Redis, SeaweedFS
	$(CONTAINER_ENGINE) compose up -d postgres redis seaweedfs
down:
	$(CONTAINER_ENGINE) compose down
migrate:                     # needs Postgres on localhost:5433 (make infra)
	cd services/api && uv run alembic upgrade head
api:                         # API on :8010
	cd services/api && uv run uvicorn wasifyai_art_api.app:app --reload --port 8010
worker:                      # job worker (needs make infra)
	cd services/api && uv run python -m wasifyai_art_api.worker
web:                         # UI on :3010
	pnpm --filter web dev
test:
	uv run pytest
	pnpm test
lint:
	uv run ruff check . && uv run ruff format --check . && uv run pyright
	pnpm lint && pnpm typecheck
format:
	uv run ruff format . && uv run ruff check --fix .
contracts:                   # export OpenAPI, regenerate TypeScript types
	uv run python scripts/export_openapi.py
	pnpm --filter @wasifyai-art/contracts generate
