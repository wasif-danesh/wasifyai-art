"""LoRA trainers: run one training job and return its checkpoints.

`MfluxTrainer` drives the `mflux-train` CLI (MLX, Apple silicon) as a subprocess;
`StubTrainer` simulates a run so the whole flow works without a GPU. Both sit behind the
`Trainer` protocol (CLAUDE.md rule 8)."""

import asyncio
import io
import json
import logging
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageDraw

from wasifyai_art_api.comfy import ComfyError, ProgressFn
from wasifyai_art_api.lora_convert import write_tensors

log = logging.getLogger("wasifyai_art_trainer")
_BAR = re.compile(r"(\d+)/(\d+)\s*\[")


@dataclass(frozen=True)
class TrainImage:
    name: str  # file stem used on disk
    png: bytes
    caption: str


@dataclass(frozen=True)
class RawCheckpoint:
    step: int
    adapter: bytes
    preview_png: bytes | None


class Trainer(Protocol):
    async def train(
        self,
        request: dict[str, Any],
        job_id: str,
        images: list[TrainImage],
        on_progress: ProgressFn,
    ) -> list[RawCheckpoint]: ...

    async def cancel(self, job_id: str) -> None: ...


def caption_with_trigger(caption: str, trigger: str) -> str:
    """The caption as trained on: the trigger word is always present (prepended when missing)."""
    c = caption.strip()
    if trigger.lower() in c.lower():
        return c
    return f"{trigger}, {c}" if c else trigger


def parse_progress(chunk: str, total: int) -> int | None:
    """The training step from a tqdm fragment, only for the bar whose total is the plan.

    The model download also prints N/M bars, which are ignored."""
    found = None
    for m in _BAR.finditer(chunk):
        if int(m.group(2)) == total:
            found = int(m.group(1))
    return found


def build_config(request: dict[str, Any], data_dir: Path, out_dir: Path) -> dict[str, Any]:
    p = request["params"]
    blocks = p["blocks"]
    targets = []
    for path, key in [
        ("transformer_blocks.{block}.attn.to_q", "double"),
        ("transformer_blocks.{block}.attn.to_k", "double"),
        ("transformer_blocks.{block}.attn.to_v", "double"),
        ("transformer_blocks.{block}.attn.to_out", "double"),
        ("transformer_blocks.{block}.attn.add_q_proj", "double"),
        ("transformer_blocks.{block}.attn.add_k_proj", "double"),
        ("transformer_blocks.{block}.attn.add_v_proj", "double"),
        ("transformer_blocks.{block}.attn.to_add_out", "double"),
        ("transformer_blocks.{block}.ff.linear_in", "double"),
        ("transformer_blocks.{block}.ff.linear_out", "double"),
        ("transformer_blocks.{block}.ff_context.linear_in", "double"),
        ("transformer_blocks.{block}.ff_context.linear_out", "double"),
        ("single_transformer_blocks.{block}.attn.to_qkv_mlp_proj", "single"),
        ("single_transformer_blocks.{block}.attn.to_out", "single"),
    ]:
        targets.append(
            {"module_path": path, "blocks": {"start": 0, "end": blocks[key]}, "rank": p["rank"]}
        )
    return {
        "model": request["model"],
        "data": str(data_dir),
        "seed": 42,
        "steps": 40,
        "guidance": 1.0,
        "quantize": p.get("quantize"),
        "low_ram": False,
        "gradient_checkpointing": p.get("gradient_checkpointing", True),
        "max_resolution": p.get("max_resolution", 768),
        "training_loop": {
            "num_epochs": request["epochs"],
            "batch_size": 1,
            # The whole noise range: composition and subject are decided at high noise.
            "timestep_low": p.get("timestep_low", 0),
            "timestep_high": p.get("timestep_high", 40),
            "timestep_type": p.get("timestep_type", "sigmoid"),
        },
        "optimizer": {"name": "AdamW", "learning_rate": p["learning_rate"]},
        "checkpoint": {"output_path": str(out_dir), "save_frequency": request["save_every"]},
        "monitoring": {"plot_frequency": 20, "generate_image_frequency": request["save_every"]},
        "lora_layers": {"targets": targets},
    }


def collect_checkpoints(out_root: Path) -> list[RawCheckpoint]:
    """Adapters (and the preview image of the same step) from mflux's timestamped output folder."""
    found: list[RawCheckpoint] = []
    for run in sorted(out_root.parent.glob(out_root.name + "_*")):
        for z in sorted((run / "checkpoints").glob("*_checkpoint.zip")):
            step = int(z.name.split("_")[0])
            if step == 0:
                continue  # the untrained state
            with zipfile.ZipFile(z) as zf:
                name = next((n for n in zf.namelist() if n.endswith("_adapter.safetensors")), None)
                if name is None:
                    continue
                adapter = zf.read(name)
            preview = next(iter(sorted((run / "preview").glob(f"{step:07d}_*.png"))), None)
            found.append(RawCheckpoint(step, adapter, preview.read_bytes() if preview else None))
    return found


class MfluxTrainer:
    def __init__(self, binary: Path, hf_disable_xet: bool = True):
        self._bin = binary
        self._env_extra = (
            {"HF_HUB_DISABLE_XET": "1", "HF_HUB_DOWNLOAD_TIMEOUT": "120"} if hf_disable_xet else {}
        )
        self._procs: dict[str, asyncio.subprocess.Process] = {}

    async def train(
        self,
        request: dict[str, Any],
        job_id: str,
        images: list[TrainImage],
        on_progress: ProgressFn,
    ) -> list[RawCheckpoint]:
        if not self._bin.is_file():
            raise ComfyError(
                "trainer_missing", "The trainer (mflux) is not installed on this machine."
            )
        total = int(request["steps"])
        work = Path(tempfile.mkdtemp(prefix="wasifyai-train-"))
        try:
            data = work / "data"
            data.mkdir()
            for img in images:
                (data / f"{img.name}.png").write_bytes(img.png)
                (data / f"{img.name}.txt").write_text(img.caption)
            (data / "preview_1.txt").write_text(images[0].caption)
            out = work / "out"
            out.mkdir()
            cfg = work / "train.json"
            cfg.write_text(json.dumps(build_config(request, data, out)))
            import os  # noqa: PLC0415

            proc = await asyncio.create_subprocess_exec(
                str(self._bin), "--config", str(cfg),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, **self._env_extra},
            )  # fmt: skip
            self._procs[job_id] = proc
            tail = ""
            assert proc.stdout is not None
            while chunk := await proc.stdout.read(4096):
                text = chunk.decode("utf-8", "replace")
                tail = (tail + text)[-4000:]
                step = parse_progress(text, total)
                if step is not None:
                    await on_progress(min(step / total, 0.99))
            code = await proc.wait()
            if job_id in self._procs and self._procs[job_id].returncode in (-15, -9):
                raise ComfyError("interrupted", "The training was stopped.")
            if code != 0:
                log.error("mflux-train exited %s: %s", code, tail[-1500:])
                raise ComfyError("training_failed", "The training failed.")
            checkpoints = await asyncio.to_thread(collect_checkpoints, out)
            if not checkpoints:
                raise ComfyError("training_failed", "The training produced no checkpoint.")
            await on_progress(1.0)
            return checkpoints
        finally:
            self._procs.pop(job_id, None)
            shutil.rmtree(work, ignore_errors=True)

    async def cancel(self, job_id: str) -> None:
        proc = self._procs.get(job_id)
        if proc and proc.returncode is None:
            proc.terminate()


class StubTrainer:
    """Simulated run: a tiny mflux-named adapter per checkpoint and placeholder previews."""

    def __init__(self, step_s: float = 0.02):
        self._step_s = step_s
        self._cancelled: set[str] = set()

    async def cancel(self, job_id: str) -> None:
        self._cancelled.add(job_id)

    async def train(
        self,
        request: dict[str, Any],
        job_id: str,
        images: list[TrainImage],
        on_progress: ProgressFn,
    ) -> list[RawCheckpoint]:
        total, every = int(request["steps"]), int(request["save_every"])
        out: list[RawCheckpoint] = []
        for step in range(1, total + 1):
            await asyncio.sleep(self._step_s)
            if job_id in self._cancelled:
                raise ComfyError("interrupted", "The training was stopped.")
            if step % max(1, total // 20) == 0:
                await on_progress(step / total)
            if step % every == 0 or step == total:
                out.append(RawCheckpoint(step, _stub_adapter(), _stub_preview(step)))
        return out


def _stub_adapter() -> bytes:
    zeros = bytes(16 * 8 * 4)
    names = [
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer.transformer_blocks.0.attn.to_out.lora_A.weight",
        "transformer.transformer_blocks.0.attn.to_out.lora_B.weight",
        "transformer.single_transformer_blocks.0.attn.to_out.lora_A.weight",
    ]
    return write_tensors(
        {n: ("F32", [16, 8], zeros) for n in names},
        {"mflux_version": "stub", "model": "flux2-klein-base-4b"},
    )


def _stub_preview(step: int) -> bytes:
    im = Image.new("RGB", (256, 256), (90 + step % 120, 120, 160))
    ImageDraw.Draw(im).text((12, 12), f"stub preview, step {step}", fill=(255, 255, 255))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()
