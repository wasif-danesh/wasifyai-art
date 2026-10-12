"""The hub's use cases: scan the model folders into the registry, and confirm a version."""

import asyncio
from pathlib import Path

from wasifyai_art_contracts import ModelKind, ScanResult

from wasifyai_art_api.families import FamilyConfig, guess_family
from wasifyai_art_api.plan import plan_scan
from wasifyai_art_api.scanner import ScannedFile, scan_models_dir
from wasifyai_art_api.store import Hint, ModelStore


def hint_for(s: ScannedFile, families: list[FamilyConfig]) -> Hint:
    meta = s.header.metadata
    stem = Path(s.rel_path).stem
    name = (meta.get("modelspec.title") or meta.get("ss_output_name") or stem).strip() or stem
    family = guess_family(
        families,
        meta.get("modelspec.architecture", ""),
        meta.get("ss_base_model_version", ""),
        meta.get("ss_sd_model_name", ""),
        stem,
    )
    return Hint(name=name[:120], family_hint=family, trigger_hints=s.header.trigger_candidates)


async def scan_into_registry(
    store: ModelStore,
    tenant_id: str,
    owner_id: str,
    root: Path,
    folders: dict[str, ModelKind],
    families: list[FamilyConfig],
) -> ScanResult:
    # File reads are header-only and fast, but they are blocking I/O: keep them off the event loop.
    outcome = await asyncio.to_thread(scan_models_dir, root, folders)
    existing = await store.list_files(tenant_id, owner_id)
    plan = plan_scan(existing, outcome.found)
    hints = {s.rel_path: hint_for(s, families) for s in outcome.found}
    await store.apply_plan(tenant_id, owner_id, plan, hints)
    return ScanResult(
        added=len(plan.add),
        updated=len(plan.update),
        missing=len(plan.missing),
        restored=len(plan.restore),
        unchanged=plan.unchanged,
        skipped_unsupported=outcome.skipped_unsupported,
        problems=outcome.problems,
    )
