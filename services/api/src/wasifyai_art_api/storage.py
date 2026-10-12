"""One object-storage interface (obstore) over a local folder or an S3-compatible service.

Keys are `{tenant}/{product}/{owner}/{path}` (CLAUDE.md rule 4). Callers pass the relative path; the
tenant prefix is added here so no caller can forget it."""

import obstore
from obstore.store import LocalStore, ObjectStore, S3Store

from wasifyai_art_api import PRODUCT_ID
from wasifyai_art_api.settings import Settings


def object_key(tenant_id: str, owner_id: str, path: str) -> str:
    if ".." in path.split("/") or path.startswith("/"):
        raise ValueError("invalid storage path")
    return f"{tenant_id}/{PRODUCT_ID}/{owner_id}/{path}"


class Storage:
    def __init__(self, store: ObjectStore):
        self._store = store

    async def put(self, tenant_id: str, owner_id: str, path: str, data: bytes) -> str:
        key = object_key(tenant_id, owner_id, path)
        await obstore.put_async(self._store, key, data)
        return path

    async def get(self, tenant_id: str, owner_id: str, path: str) -> bytes:
        res = await obstore.get_async(self._store, object_key(tenant_id, owner_id, path))
        return bytes(await res.bytes_async())

    async def delete(self, tenant_id: str, owner_id: str, path: str) -> None:
        try:
            await obstore.delete_async(self._store, object_key(tenant_id, owner_id, path))
        except FileNotFoundError:
            pass


def make_storage(settings: Settings) -> Storage:
    if settings.storage_backend == "s3":
        store: ObjectStore = S3Store(
            settings.storage_bucket,
            endpoint=settings.storage_endpoint,
            access_key_id=settings.storage_access_key,
            secret_access_key=settings.storage_secret_key,
            region="us-east-1",
            client_options={"allow_http": settings.storage_endpoint.startswith("http://")},
            virtual_hosted_style_request=False,
        )
    else:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        store = LocalStore(settings.data_dir)
    return Storage(store)


class InMemoryStorage(Storage):
    """For tests: same interface, a dict."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put(self, tenant_id: str, owner_id: str, path: str, data: bytes) -> str:
        self.objects[object_key(tenant_id, owner_id, path)] = data
        return path

    async def get(self, tenant_id: str, owner_id: str, path: str) -> bytes:
        try:
            return self.objects[object_key(tenant_id, owner_id, path)]
        except KeyError:
            raise FileNotFoundError(path) from None

    async def delete(self, tenant_id: str, owner_id: str, path: str) -> None:
        self.objects.pop(object_key(tenant_id, owner_id, path), None)
