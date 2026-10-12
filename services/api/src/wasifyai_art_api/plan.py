"""Decide what a scan changes. Pure: no I/O, so it is tested without a database."""

from dataclasses import dataclass, field

from wasifyai_art_api.scanner import ScannedFile


@dataclass(frozen=True)
class StoredFile:
    file_id: str
    version_id: str
    rel_path: str
    file_status: str  # present | missing
    header_sha256: str
    size_bytes: int


@dataclass
class Plan:
    add: list[ScannedFile] = field(default_factory=list)
    # changed content under a known path: the version needs confirming again
    update: list[tuple[StoredFile, ScannedFile]] = field(default_factory=list)
    missing: list[StoredFile] = field(default_factory=list)
    # (stored, scanned, same_content)
    restore: list[tuple[StoredFile, ScannedFile, bool]] = field(default_factory=list)
    unchanged: int = 0


def plan_scan(existing: list[StoredFile], found: list[ScannedFile]) -> Plan:
    by_path = {e.rel_path: e for e in existing}
    seen: set[str] = set()
    plan = Plan()
    for s in found:
        seen.add(s.rel_path)
        e = by_path.get(s.rel_path)
        if e is None:
            plan.add.append(s)
            continue
        same = e.header_sha256 == s.header.header_sha256 and e.size_bytes == s.size_bytes
        if e.file_status == "missing":
            plan.restore.append((e, s, same))
        elif same:
            plan.unchanged += 1
        else:
            plan.update.append((e, s))
    plan.missing = [e for e in existing if e.rel_path not in seen and e.file_status == "present"]
    return plan
