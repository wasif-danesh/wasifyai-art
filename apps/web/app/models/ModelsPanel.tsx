"use client";

import { formatSize, parseWords } from "@/lib/format";
import type {
  FamilyInfo,
  ModelSummary,
  ModelVersionSummary,
  ScanResult,
} from "@wasifyai-art/contracts";
import { useState } from "react";

const STATUS_STYLE: Record<string, string> = {
  ready: "text-green-600",
  unconfirmed: "text-amber-600",
  missing: "text-red-600",
  disabled: "text-[var(--muted)]",
};

export function ModelsPanel({
  initialModels,
  families,
}: {
  initialModels: ModelSummary[];
  families: FamilyInfo[];
}) {
  const [models, setModels] = useState(initialModels);
  const [kind, setKind] = useState<"all" | "lora" | "checkpoint">("all");
  const [scan, setScan] = useState<ScanResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    const r = await fetch("/api/models", { cache: "no-store" });
    if (r.ok) setModels(((await r.json()) as { models: ModelSummary[] }).models);
  }

  async function runScan() {
    setBusy(true);
    setError(null);
    try {
      const r = await fetch("/api/models/scan", { method: "POST" });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail ?? `Scan failed (${r.status})`);
      setScan(body as ScanResult);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Scan failed");
    } finally {
      setBusy(false);
    }
  }

  const shown = models.filter((m) => kind === "all" || m.kind === kind);

  return (
    <div className="space-y-6">
      <header className="flex items-end justify-between">
        <div>
          <h1 className="text-xl font-medium">Models</h1>
          <p className="text-sm text-[var(--muted)]">
            Checkpoints and LoRAs found in your model folders. Confirm the base family before use.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <select
            aria-label="Filter by kind"
            value={kind}
            onChange={(e) => setKind(e.target.value as typeof kind)}
            className="rounded border border-[var(--border)] bg-transparent px-2 py-1 text-sm"
          >
            <option value="all">All</option>
            <option value="lora">LoRAs</option>
            <option value="checkpoint">Checkpoints</option>
          </select>
          <button
            type="button"
            onClick={runScan}
            disabled={busy}
            className="rounded bg-[var(--fg)] px-3 py-1.5 text-sm text-[var(--bg)] disabled:opacity-60"
          >
            {busy ? "Scanning…" : "Scan folders"}
          </button>
        </div>
      </header>

      {error && (
        <p role="alert" className="rounded border border-red-600 px-3 py-2 text-sm text-red-600">
          {error}
        </p>
      )}
      {scan && (
        <output className="block text-sm text-[var(--muted)]">
          {scan.added} added, {scan.updated} changed, {scan.missing} missing, {scan.restored}{" "}
          restored, {scan.unchanged} unchanged.
          {scan.problems.length > 0 && ` ${scan.problems.length} could not be read.`}
          {scan.skipped_unsupported.length > 0 &&
            ` ${scan.skipped_unsupported.length} skipped (only .safetensors is supported).`}
        </output>
      )}

      {shown.length === 0 ? (
        <p className="text-sm text-[var(--muted)]">
          No models yet. Set MODELS_DIR to your ComfyUI models folder and scan.
        </p>
      ) : (
        <ul className="divide-y divide-[var(--border)] rounded border border-[var(--border)]">
          {shown.map((m) => (
            <li key={m.id} className="space-y-2 px-4 py-3">
              <div className="flex items-baseline justify-between">
                <span className="font-medium">{m.name}</span>
                <span className="text-xs uppercase tracking-wide text-[var(--muted)]">
                  {m.kind}
                </span>
              </div>
              {m.versions.map((v) => (
                <VersionRow key={v.id} version={v} families={families} onChanged={refresh} />
              ))}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function VersionRow({
  version,
  families,
  onChanged,
}: {
  version: ModelVersionSummary;
  families: FamilyInfo[];
  onChanged: () => Promise<void>;
}) {
  const [family, setFamily] = useState(version.family ?? version.family_hint ?? "");
  const [words, setWords] = useState(
    (version.trigger_words.length ? version.trigger_words : version.trigger_hints).join(", "),
  );
  const [error, setError] = useState<string | null>(null);
  const file = version.files[0];

  async function save(status: "ready" | "disabled" | "unconfirmed") {
    setError(null);
    const r = await fetch(`/api/model-versions/${version.id}`, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ family: family || null, trigger_words: parseWords(words), status }),
    });
    if (!r.ok) {
      setError(((await r.json()) as { detail?: string }).detail ?? "Could not save");
      return;
    }
    await onChanged();
  }

  return (
    <div className="grid gap-2 text-sm md:grid-cols-[1fr_auto]">
      <div className="space-y-1">
        <div className="flex gap-3 text-[var(--muted)]">
          <span className={STATUS_STYLE[version.status]}>{version.status}</span>
          {file && (
            <span>
              {file.rel_path} · {formatSize(file.size_bytes)}
            </span>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <label className="sr-only" htmlFor={`fam-${version.id}`}>
            Base family
          </label>
          <select
            id={`fam-${version.id}`}
            value={family}
            onChange={(e) => setFamily(e.target.value)}
            className="rounded border border-[var(--border)] bg-transparent px-2 py-1"
          >
            <option value="">Family: not set</option>
            {families.map((f) => (
              <option key={f.id} value={f.id}>
                {f.title}
                {f.id === version.family_hint && !version.family ? " (suggested)" : ""}
              </option>
            ))}
          </select>
          <label className="sr-only" htmlFor={`tw-${version.id}`}>
            Trigger words
          </label>
          <input
            id={`tw-${version.id}`}
            value={words}
            onChange={(e) => setWords(e.target.value)}
            placeholder="Trigger words, comma separated"
            className="min-w-56 flex-1 rounded border border-[var(--border)] bg-transparent px-2 py-1"
          />
        </div>
        {error && (
          <p role="alert" className="text-red-600">
            {error}
          </p>
        )}
      </div>
      <div className="flex items-start gap-2">
        <button
          type="button"
          onClick={() => save("ready")}
          className="rounded border border-[var(--border)] px-2 py-1"
        >
          {version.status === "ready" ? "Save" : "Mark ready"}
        </button>
        <button
          type="button"
          onClick={() => save(version.status === "disabled" ? "unconfirmed" : "disabled")}
          className="rounded border border-[var(--border)] px-2 py-1 text-[var(--muted)]"
        >
          {version.status === "disabled" ? "Enable" : "Disable"}
        </button>
      </div>
    </div>
  );
}
