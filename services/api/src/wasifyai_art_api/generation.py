"""Turn a user's request into a fully resolved job request, or a plain refusal."""

import secrets
from typing import Any

from wasifyai_art_contracts import GenerationRequest, ProviderInfo

from wasifyai_art_api.families import FamilyConfig
from wasifyai_art_api.jobs import JobStore
from wasifyai_art_api.store import ModelStore
from wasifyai_art_api.uploads import UploadStore
from wasifyai_art_api.workflows import WorkflowRegistry


class GenerationRefused(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


async def resolve_request(
    body: GenerationRequest,
    tenant_id: str,
    owner_id: str,
    store: ModelStore,
    jobs: JobStore,
    uploads: UploadStore,
    providers: list[ProviderInfo],
    families: list[FamilyConfig],
    workflows: WorkflowRegistry,
    weight_range: tuple[float, float] = (-1.0, 2.0),
) -> dict[str, Any]:
    provider = next((p for p in providers if p.id == body.provider_id), None)
    if provider is None:
        raise GenerationRefused(422, f"unknown provider {body.provider_id!r}")
    if provider.status != "configured":
        raise GenerationRefused(409, f"{provider.title} is not configured")
    edit = body.mode == "edit"
    if edit and not provider.capabilities.edit:
        raise GenerationRefused(409, f"{provider.title} cannot edit pictures")
    if not edit and not provider.capabilities.text2img:
        raise GenerationRefused(409, f"{provider.title} cannot make images from text")

    # The creation this one is built from: its own picture (edit) or only a lineage link (remix)
    parent_id = body.source_creation_id if edit else body.parent_id
    parent = None
    if parent_id:
        parent = await jobs.get_creation(tenant_id, owner_id, parent_id)
        if parent is None:
            raise GenerationRefused(404, "the creation to remix was not found")
    upload = None
    if edit and body.source_upload_id:
        upload = await uploads.get(tenant_id, owner_id, body.source_upload_id)
        if upload is None or upload.consumed:
            raise GenerationRefused(404, "the uploaded picture was not found: upload it again")

    version = await store.get_version_detail(tenant_id, owner_id, body.version_id)
    if version is None:
        raise GenerationRefused(404, "model version not found")
    if version.kind != "checkpoint":
        raise GenerationRefused(422, "choose a checkpoint as the base model")
    if version.status != "ready" or version.rel_path is None:
        raise GenerationRefused(409, "this model is not ready: confirm its base family first")
    family = next((f for f in families if f.id == version.family), None)
    if family is None or family.generation is None:
        raise GenerationRefused(409, f"text to image is not set up for family {version.family!r}")
    capability = "edit" if edit else "text2img"
    workflow = workflows.find(capability, family.id)
    if workflow is None:
        raise GenerationRefused(409, f"no workflow is installed for family {family.id!r}")
    if edit:
        # the result keeps the picture's shape (the workflow scales to about one megapixel)
        source = parent or upload
        assert source is not None
        size = (source.width, source.height)
        steps = family.generation.edit_steps or family.generation.steps
    else:
        sized = family.generation.sizes.get(body.size)
        if sized is None:
            raise GenerationRefused(422, f"size {body.size!r} is not available for {family.title}")
        size, steps = sized, family.generation.steps

    lora_requests: list[dict[str, Any]] = []
    if body.loras:
        chain = workflow.map.lora_chain
        if not provider.capabilities.loras or chain is None:
            raise GenerationRefused(409, f"{provider.title} cannot apply LoRAs to {family.title}")
        limit = min(provider.capabilities.lora_stack_max, chain.max)
        if len(body.loras) > limit:
            raise GenerationRefused(422, f"at most {limit} LoRAs can be used at once")
        if len({u.version_id for u in body.loras}) != len(body.loras):
            raise GenerationRefused(422, "each LoRA can be used once")
        for use in body.loras:
            if not weight_range[0] <= use.weight <= weight_range[1]:
                raise GenerationRefused(
                    422, f"LoRA weight must be between {weight_range[0]} and {weight_range[1]}"
                )
            lv = await store.get_version_detail(tenant_id, owner_id, use.version_id)
            if lv is None:
                raise GenerationRefused(404, "LoRA version not found")
            if lv.kind != "lora":
                raise GenerationRefused(422, f"{lv.name} is not a LoRA")
            if lv.status != "ready" or lv.rel_path is None:
                raise GenerationRefused(
                    409, f"{lv.name} is not ready: confirm its base family first"
                )
            if lv.family != family.id:
                raise GenerationRefused(
                    422, f"{lv.name} is for {lv.family}, not {family.id}: it cannot be used here"
                )
            lora_requests.append(
                {
                    "version_id": lv.version_id,
                    "name": lv.name,
                    "rel_path": lv.rel_path,
                    "header_sha256": lv.header_sha256,
                    "file": _comfy_name(lv.rel_path),
                    "weight": use.weight,
                }  # fmt: skip
            )

    model_file = _comfy_name(version.rel_path)
    source_key: str | None = None
    if edit:
        source_key = parent.image_key if parent else (upload.key if upload else None)
    return {
        "provider_id": provider.id,
        "version_id": version.version_id,
        "model_name": version.name,
        "family": family.id,
        "rel_path": version.rel_path,
        "header_sha256": version.header_sha256,
        "model_file": model_file,
        "loras": lora_requests,
        "workflow": workflow.name,
        "prompt": body.prompt.strip(),
        "size": body.size,
        "width": size[0],
        "height": size[1],
        "steps": steps,
        "mode": body.mode,
        "parent_id": parent_id,
        "source_key": source_key,
        "source_upload_id": upload.id if upload else None,
        "seed": body.seed if body.seed is not None else secrets.randbelow(2**31),
    }


def _comfy_name(rel_path: str) -> str:
    """ComfyUI lists a model by its path inside the model folder, without the folder's own name."""
    return rel_path.split("/", 1)[1] if "/" in rel_path else rel_path
