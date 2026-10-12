"""ASGI entry point: `uvicorn wasifyai_art_api.app:app`."""

from wasifyai_art_api.main import create_app

app = create_app()
