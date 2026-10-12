"""Wire contracts: the Pydantic models are the source of truth.

TypeScript types are generated from the API's OpenAPI schema (`make contracts`).
Never hand-edit the generated file."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ModelKind = Literal["checkpoint", "lora"]
Visibility = Literal["private", "unlisted", "public"]
VersionStatus = Literal["unconfirmed", "ready", "missing", "disabled"]
ProviderStatus = Literal["configured", "unconfigured"]


class Out(BaseModel):
    """Base of every response model.

    Fields with defaults are still always present on the wire, so the generated TypeScript types
    mark them required."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


class Health(Out):
    status: Literal["ok"] = "ok"
    product: str


class ProviderCapabilities(Out):
    """What a provider can do; the editor shows only controls the chosen provider supports."""

    text2img: bool = True
    img2img: bool = False
    edit: bool = False
    video: bool = False
    loras: bool = False
    lora_stack_max: int = 0
    negative_prompt: bool = False


class ProviderInfo(Out):
    id: str
    kind: str
    title: str
    status: ProviderStatus
    capabilities: ProviderCapabilities


class ProviderList(Out):
    providers: list[ProviderInfo]


class ModelFileSummary(Out):
    rel_path: str
    folder: str
    size_bytes: int
    status: Literal["present", "missing"]


class ModelVersionSummary(Out):
    id: str
    label: str
    family: str | None = None
    status: VersionStatus
    trigger_words: list[str] = Field(default_factory=list)
    # Suggestions read from the file's own metadata; never trusted until the owner confirms.
    family_hint: str | None = None
    trigger_hints: list[str] = Field(default_factory=list)
    files: list[ModelFileSummary] = Field(default_factory=list)


class ModelSummary(Out):
    id: str
    kind: ModelKind
    name: str
    slug: str
    tags: list[str] = Field(default_factory=list)
    visibility: Visibility = "private"
    versions: list[ModelVersionSummary] = Field(default_factory=list)


class ModelList(Out):
    models: list[ModelSummary]


class ScanProblem(Out):
    rel_path: str
    reason: str


class ScanResult(Out):
    added: int = 0
    updated: int = 0
    missing: int = 0
    restored: int = 0
    unchanged: int = 0
    skipped_unsupported: list[str] = Field(default_factory=list)
    problems: list[ScanProblem] = Field(default_factory=list)


class VersionUpdate(BaseModel):
    """Owner's confirmation of a version. `ready` needs a family."""

    family: str | None = None
    trigger_words: list[str] | None = None
    status: Literal["unconfirmed", "ready", "disabled"] | None = None


class FamilyInfo(Out):
    id: str
    title: str


class FamilyList(Out):
    families: list[FamilyInfo]


SizePreset = Literal["square", "landscape", "portrait", "wide", "tall"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "canceled"]


class LoraUse(BaseModel):
    version_id: str
    weight: float = Field(default=1.0, ge=-2.0, le=3.0)


class GenerationRequest(BaseModel):
    """`text`: a prompt and a size. `edit`: one of the caller's own creations plus an instruction
    (image to image). `parent_id` records the creation this one is a remix of."""

    provider_id: str
    version_id: str
    prompt: str = Field(min_length=1, max_length=500)
    mode: Literal["text", "edit"] = "text"
    size: SizePreset = "square"
    seed: int | None = Field(default=None, ge=0, lt=2**31)
    source_creation_id: str | None = None
    source_upload_id: str | None = None
    parent_id: str | None = None
    loras: list[LoraUse] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def _mode_fields(self) -> "GenerationRequest":
        sources = [s for s in (self.source_creation_id, self.source_upload_id) if s]
        if self.mode == "edit" and len(sources) != 1:
            raise ValueError("edit needs exactly one of source_creation_id and source_upload_id")
        if self.mode == "text" and sources:
            raise ValueError("a source picture is only for edit")
        return self


class JobSummary(Out):
    id: str
    status: JobStatus
    progress: float
    prompt: str
    mode: str
    model_name: str
    error: str | None = None
    creation_id: str | None = None
    created_at: str


class JobList(Out):
    jobs: list["JobSummary"]


class CreationModelRef(Out):
    version_id: str | None
    role: Literal["base", "lora"] = "base"
    weight: float | None = None
    name: str
    family: str | None
    rel_path: str
    header_sha256: str | None


class CreationSummary(Out):
    id: str
    prompt: str
    width: int
    height: int
    provider_id: str
    created_at: str
    parent_id: str | None
    image_url: str  # a route of this API, checked against the owner on every fetch
    thumb_url: str
    params: dict[str, str | int | float | bool | None]
    models: list[CreationModelRef]


class CreationList(Out):
    creations: list[CreationSummary]


class UploadSummary(Out):
    upload_id: str
    width: int
    height: int
    bytes: int
