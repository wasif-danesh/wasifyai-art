"""Provider adapters: run one resolved generation request and return the image bytes.

The API resolves everything (workflow, file, size, seed) when it accepts a request,
so a runner only executes."""

import asyncio
import io
from typing import Any, Protocol

from PIL import Image, ImageDraw

from wasifyai_art_api.comfy import ComfyClient, ComfyError, ProgressFn
from wasifyai_art_api.workflows import WorkflowRegistry


class Runner(Protocol):
    async def run(
        self,
        request: dict[str, Any],
        job_id: str,
        on_progress: ProgressFn,
        inputs: dict[str, bytes] | None = None,
    ) -> bytes: ...

    async def cancel(self, job_id: str) -> None: ...


class ComfyRunner:
    def __init__(self, client: ComfyClient, workflows: WorkflowRegistry, timeout_s: float):
        self._client = client
        self._workflows = workflows
        self._timeout = timeout_s

    async def run(
        self,
        request: dict[str, Any],
        job_id: str,
        on_progress: ProgressFn,
        inputs: dict[str, bytes] | None = None,
    ) -> bytes:
        wf = self._workflows.get(request["workflow"])
        if wf is None:
            raise ComfyError("invalid_workflow", "This workflow is not installed.")
        values: dict[str, Any] = {
            "model": request["model_file"],
            "prompt": request["prompt"],
            "seed": request["seed"],
            "steps": request["steps"],
            "width": request["width"],
            "height": request["height"],
        }
        if request.get("mode") == "edit":
            picture = (inputs or {}).get("image")
            if picture is None:
                raise ComfyError("invalid_input", "The source picture is missing.")
            values["image"] = await self._client.upload_image(f"wsh-{job_id}.png", picture)
        # only inputs this workflow declares (an edit workflow has no size)
        loras = [(x["file"], float(x["weight"])) for x in request.get("loras", [])]
        graph = wf.fill({k: v for k, v in values.items() if k in wf.map.inputs}, loras or None)
        outputs = await self._client.run(graph, job_id, on_progress, self._timeout)
        node = wf.map.outputs["image"].node
        data = await self._client.fetch(self._client.pick_image(outputs, node))
        await self._client.free()
        return data

    async def cancel(self, job_id: str) -> None:
        # ComfyUI's interrupt applies to the running prompt; the GPU lock means that is this job's.
        await self._client.interrupt()


class StubRunner:
    """Placeholder image and simulated progress: the whole flow works without ComfyUI or a GPU."""

    def __init__(self, step_s: float = 0.15) -> None:
        self._step_s = step_s
        self._cancelled: set[str] = set()

    async def cancel(self, job_id: str) -> None:
        self._cancelled.add(job_id)

    async def run(
        self,
        request: dict[str, Any],
        job_id: str,
        on_progress: ProgressFn,
        inputs: dict[str, bytes] | None = None,
    ) -> bytes:
        for i in range(1, 5):
            await asyncio.sleep(self._step_s)
            if job_id in self._cancelled:
                raise ComfyError("interrupted", "The generation was interrupted.")
            await on_progress(i / 4)
        w, h = int(request["width"]), int(request["height"])
        if request.get("mode") == "edit" and (inputs or {}).get("image"):
            with Image.open(io.BytesIO(inputs["image"])) as src:  # type: ignore[index]
                w, h = src.size
        seed = int(request["seed"])
        img = Image.new("RGB", (w, h), ((seed * 37) % 200 + 30, (seed * 91) % 200 + 30, 120))
        draw = ImageDraw.Draw(img)
        draw.text((24, 24), f"stub  seed {seed}\n{request['prompt'][:80]}", fill=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()
