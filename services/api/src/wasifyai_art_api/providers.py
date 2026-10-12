"""Provider catalogue from config/providers.yaml.

A provider is `configured` when its address variable is set."""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel
from wasifyai_art_contracts import ProviderCapabilities, ProviderInfo


class ProviderConfig(BaseModel):
    id: str
    kind: str
    title: str
    url_env: str | None = None
    capabilities: ProviderCapabilities = ProviderCapabilities()


def load_providers(path: Path) -> list[ProviderConfig]:
    data = yaml.safe_load(path.read_text()) or {}
    return [ProviderConfig.model_validate(p) for p in data.get("providers", [])]


def describe(
    configs: list[ProviderConfig], env: dict[str, str] | None = None
) -> list[ProviderInfo]:
    env = dict(os.environ) if env is None else env
    return [
        ProviderInfo(
            id=c.id,
            kind=c.kind,
            title=c.title,
            status="configured" if c.url_env and env.get(c.url_env) else "unconfigured",
            capabilities=c.capabilities,
        )
        for c in configs
    ]
