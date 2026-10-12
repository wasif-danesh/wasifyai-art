"""ComfyUI client: POST /prompt, progress over the WebSocket, outputs from /history and /view."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
import websockets

log = logging.getLogger(__name__)
ProgressFn = Callable[[float], Awaitable[None]]


class ComfyError(Exception):
    """A failure of the job itself; `message` is safe to show the user."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ComfyFile:
    filename: str
    subfolder: str
    type: str


class ComfyClient:
    def __init__(self, base_url: str, http: httpx.AsyncClient | None = None):
        self._base = base_url.rstrip("/")
        self._http = http or httpx.AsyncClient(timeout=60)

    @property
    def _ws_url(self) -> str:
        return self._base.replace("https://", "wss://").replace("http://", "ws://")

    async def run(
        self, graph: dict[str, Any], client_id: str, on_progress: ProgressFn, timeout_s: float
    ) -> dict[str, Any]:
        # Connect first so no progress message is missed between submitting and subscribing.
        async with websockets.connect(
            f"{self._ws_url}/ws?clientId={client_id}", max_size=None
        ) as ws:
            prompt_id = await self._submit(graph, client_id)
            try:
                async with asyncio.timeout(timeout_s):
                    await self._follow(ws, prompt_id, on_progress)
            except TimeoutError:
                await self.interrupt()
                raise
        return await self._outputs(prompt_id)

    async def _submit(self, graph: dict[str, Any], client_id: str) -> str:
        r = await self._http.post(
            f"{self._base}/prompt", json={"prompt": graph, "client_id": client_id}
        )
        if r.status_code == 400:
            log.error("comfyui rejected the workflow: %s", r.text[:600])
            raise ComfyError("invalid_workflow", "The generation request was rejected by ComfyUI.")
        r.raise_for_status()
        return r.json()["prompt_id"]

    async def _follow(self, ws: Any, prompt_id: str, on_progress: ProgressFn) -> None:
        async for raw in ws:
            if isinstance(raw, bytes):  # preview frames
                continue
            msg = json.loads(raw)
            data = msg.get("data", {})
            if data.get("prompt_id") not in (None, prompt_id):
                continue
            match msg.get("type"):
                case "progress" if data.get("max"):
                    await on_progress(min(data["value"] / data["max"], 1.0))
                case "executing" if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    return
                case "execution_error":
                    log.error("comfyui execution error: %s", data.get("exception_message"))
                    raise ComfyError("execution_failed", "The generation failed in ComfyUI.")
                case "execution_interrupted":
                    raise ComfyError("interrupted", "The generation was interrupted.")
        raise ComfyError("connection_lost", "Lost the connection to ComfyUI.")

    async def _outputs(self, prompt_id: str) -> dict[str, Any]:
        r = await self._http.get(f"{self._base}/history/{prompt_id}")
        r.raise_for_status()
        entry = r.json().get(prompt_id, {})
        if entry.get("status", {}).get("status_str") == "error":
            raise ComfyError("execution_failed", "The generation failed in ComfyUI.")
        return entry.get("outputs", {})

    @staticmethod
    def pick_image(outputs: dict[str, Any], node: str) -> ComfyFile:
        files = outputs.get(node, {}).get("images") or []
        if not files:
            raise ComfyError("missing_output", "The workflow produced no image.")
        f = files[0]
        return ComfyFile(f["filename"], f.get("subfolder", ""), f.get("type", "output"))

    async def fetch(self, f: ComfyFile) -> bytes:
        r = await self._http.get(
            f"{self._base}/view",
            params={"filename": f.filename, "subfolder": f.subfolder, "type": f.type},
        )
        r.raise_for_status()
        return r.content

    async def upload_image(self, name: str, data: bytes) -> str:
        """Put a picture in ComfyUI's temp folder (which it empties on restart) and return the
        reference `LoadImage` reads."""
        r = await self._http.post(
            f"{self._base}/upload/image",
            data={"type": "temp", "overwrite": "true"},
            files={"image": (name, data, "image/png")},
        )
        r.raise_for_status()
        return f"{r.json().get('name', name)} [temp]"

    async def free(self) -> None:
        try:
            await self._http.post(
                f"{self._base}/free", json={"unload_models": True, "free_memory": True}
            )
        except httpx.HTTPError:
            log.warning("could not ask ComfyUI to free memory", exc_info=True)

    async def interrupt(self) -> None:
        try:
            await self._http.post(f"{self._base}/interrupt")
        except httpx.HTTPError:
            log.warning("could not interrupt ComfyUI", exc_info=True)
