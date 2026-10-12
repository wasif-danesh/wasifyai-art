from dataclasses import dataclass

from fastapi import Depends

from wasifyai_art_api.settings import Settings, get_app_settings


@dataclass(frozen=True)
class Identity:
    tenant_id: str
    user_id: str


def resolve_identity(settings: Settings = Depends(get_app_settings)) -> Identity:
    """The signed-in user. `stub` is for local development and tests only; a real mode must be added
    before this runs anywhere shared (the application refuses to start with any other AUTH_MODE)."""
    return Identity(tenant_id=settings.stub_tenant_id, user_id=settings.stub_user_id)
