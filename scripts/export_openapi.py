"""Write the API's OpenAPI schema for the TypeScript type generator (`make contracts`)."""

import json
from pathlib import Path

from wasifyai_art_api.main import create_app
from wasifyai_art_api.settings import Settings
from wasifyai_art_api.store import InMemoryModelStore

OUT = Path(__file__).resolve().parents[1] / "packages" / "contracts-ts" / "openapi.json"

app = create_app(Settings(), InMemoryModelStore())
OUT.write_text(json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n")
print(f"wrote {OUT}")
