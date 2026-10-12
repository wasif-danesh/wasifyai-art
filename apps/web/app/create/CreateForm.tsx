"use client";

import { parseJobEvent } from "@/lib/sse";
import type { ProviderInfo } from "@wasifyai-art/contracts";
import Link from "next/link";
import { useRef, useState } from "react";
import { type Picked, PictureInput } from "./PictureInput";

type Base = { id: string; name: string; family: string | null };
const SIZES = ["square", "landscape", "portrait", "wide", "tall"] as const;
type Size = (typeof SIZES)[number];

/** What a remix starts from: a creation's settings (text) or its picture (edit). */
export type Initial = {
  mode: "text" | "edit";
  prompt: string;
  versionId: string | null;
  modelName: string;
  size: string;
  seed: number | null;
  sourceId: string;
  loras: { versionId: string; name: string; weight: number }[];
};

type Lora = { id: string; name: string; family: string | null; triggerWords: string[] };
type Used = { versionId: string; weight: number };

type Phase =
  | { name: "idle" }
  | { name: "working"; label: string; progress: number }
  | { name: "done"; creationId: string }
  | { name: "failed"; message: string };

export function CreateForm({
  providers,
  bases,
  loras,
  initial,
}: {
  providers: ProviderInfo[];
  bases: Base[];
  loras: Lora[];
  initial?: Initial;
}) {
  const usable = providers.filter((p) => p.status === "configured" && p.capabilities.text2img);
  const [providerId, setProviderId] = useState(usable[0]?.id ?? "");
  const inherited = initial?.versionId && bases.some((b) => b.id === initial.versionId);
  const [versionId, setVersionId] = useState(
    inherited ? (initial?.versionId as string) : (bases[0]?.id ?? ""),
  );
  // a picture edit needs an instruction ("make it dusk"), not the scene's description
  const [prompt, setPrompt] = useState(initial?.mode === "edit" ? "" : (initial?.prompt ?? ""));
  const [size, setSize] = useState<Size>(
    (SIZES as readonly string[]).includes(initial?.size ?? "") ? (initial?.size as Size) : "square",
  );
  const [keepSeed, setKeepSeed] = useState(false);
  const family = bases.find((b) => b.id === versionId)?.family ?? null;
  const fits = (l: Lora, fam: string | null) => fam !== null && l.family === fam;
  // LoRAs from the remix that are still ready and fit the base model
  const carried = (initial?.loras ?? []).filter((x) =>
    loras.some((l) => l.id === x.versionId && fits(l, family)),
  );
  const lost = (initial?.loras ?? []).filter(
    (x) => !carried.some((c) => c.versionId === x.versionId),
  );
  const [stack, setStack] = useState<Used[]>(
    carried.map((x) => ({ versionId: x.versionId, weight: x.weight })),
  );
  const [note, setNote] = useState<string | null>(null);
  const available = loras.filter(
    (l) => fits(l, family) && !stack.some((s) => s.versionId === l.id),
  );

  function changeBase(id: string) {
    const fam = bases.find((b) => b.id === id)?.family ?? null;
    const keep = stack.filter((s) => loras.some((l) => l.id === s.versionId && fits(l, fam)));
    setNote(
      keep.length < stack.length
        ? `${stack.length - keep.length} LoRA(s) were removed: they do not fit this base model.`
        : null,
    );
    setStack(keep);
    setVersionId(id);
  }

  function addTrigger(word: string) {
    setPrompt((p) =>
      p.toLowerCase().includes(word.toLowerCase()) ? p : p ? `${p}, ${word}` : word,
    );
  }
  // a fresh Create can start from text or from an uploaded picture; a remix is fixed by its source
  const [fresh, setFresh] = useState<"text" | "picture">("text");
  const [picked, setPicked] = useState<Picked | null>(null);
  const mode = initial?.mode ?? (fresh === "picture" ? "edit" : "text");
  const [phase, setPhase] = useState<Phase>({ name: "idle" });
  const [jobId, setJobId] = useState<string | null>(null);
  const needsPicture = mode === "edit" && !initial && !picked;
  const source = useRef<EventSource | null>(null);

  const blocked =
    usable.length === 0
      ? "No provider is configured. Set COMFYUI_BASE_URL (or COMFYUI_MODE=stub) and restart the API."
      : bases.length === 0
        ? "No model is ready. Scan your folders and confirm a base family on the Models page."
        : null;

  async function generate(e: React.FormEvent) {
    e.preventDefault();
    if (!prompt.trim()) return;
    setPhase({ name: "working", label: "Queued", progress: 0 });
    const r = await fetch("/api/generations", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(
        mode === "edit"
          ? {
              provider_id: providerId,
              version_id: versionId,
              prompt,
              loras: stack.map((s) => ({ version_id: s.versionId, weight: s.weight })),
              mode: "edit",
              ...(initial
                ? { source_creation_id: initial.sourceId }
                : { source_upload_id: picked?.uploadId }),
              ...(keepSeed && initial?.seed != null ? { seed: initial.seed } : {}),
            }
          : {
              provider_id: providerId,
              version_id: versionId,
              prompt,
              size,
              loras: stack.map((s) => ({ version_id: s.versionId, weight: s.weight })),
              ...(initial ? { parent_id: initial.sourceId } : {}),
              ...(keepSeed && initial?.seed != null ? { seed: initial.seed } : {}),
            },
      ),
    });
    const body = await r.json();
    if (!r.ok) {
      const detail = Array.isArray(body.detail) ? "Check the form and try again." : body.detail;
      setPhase({ name: "failed", message: detail ?? "Could not start the generation" });
      return;
    }
    source.current?.close();
    setJobId(body.id);
    const es = new EventSource(`/api/jobs/${body.id}/events`);
    source.current = es;
    for (const type of ["queued", "running", "progress", "succeeded", "failed", "canceled"]) {
      es.addEventListener(type, (ev) => {
        const parsed = parseJobEvent(type, (ev as MessageEvent).data);
        if (!parsed) return;
        if (parsed.type === "queued") setPhase({ name: "working", label: "Queued", progress: 0 });
        if (parsed.type === "running")
          setPhase({ name: "working", label: "Generating", progress: 0 });
        if (parsed.type === "progress")
          setPhase({ name: "working", label: "Generating", progress: parsed.progress });
        if (parsed.type === "succeeded") {
          setPhase({ name: "done", creationId: parsed.creationId });
          es.close();
        }
        if (parsed.type === "failed") {
          setPhase({ name: "failed", message: parsed.message });
          es.close();
        }
        if (parsed.type === "canceled") {
          setPhase({ name: "failed", message: "Canceled." });
          es.close();
        }
      });
    }
  }

  const working = phase.name === "working";

  async function cancel() {
    if (jobId) await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
  }
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-medium">
        {mode === "edit" ? "Remix picture" : initial ? "Remix settings" : "Create"}
      </h1>
      {initial && (
        <div className="flex items-center gap-3 text-sm text-[var(--muted)]">
          <img
            src={`/api/creations/${initial.sourceId}/thumb`}
            alt=""
            className="h-16 w-16 rounded border border-[var(--border)] object-cover"
          />
          <span>
            Remixing{" "}
            <Link href={`/creations/${initial.sourceId}`} className="underline">
              a creation
            </Link>
            .{" "}
            <Link href="/create" className="underline">
              Start fresh
            </Link>
          </span>
        </div>
      )}
      {initial && !inherited && (
        <p
          role="alert"
          className="rounded border border-amber-600 px-3 py-2 text-sm text-amber-600"
        >
          {initial.modelName} is not ready any more, so another base model is selected. Results will
          differ from the original.
        </p>
      )}
      {blocked && (
        <p
          role="alert"
          className="rounded border border-amber-600 px-3 py-2 text-sm text-amber-600"
        >
          {blocked}
        </p>
      )}
      {!initial && (
        <div className="flex gap-2 text-sm" role="tablist" aria-label="Start from">
          {(["text", "picture"] as const).map((k) => (
            <button
              key={k}
              type="button"
              role="tab"
              aria-selected={fresh === k}
              onClick={() => setFresh(k)}
              className={`rounded border px-3 py-1 ${fresh === k ? "border-[var(--fg)]" : "border-[var(--border)] text-[var(--muted)]"}`}
            >
              {k === "text" ? "Text to image" : "From a picture"}
            </button>
          ))}
        </div>
      )}
      <form onSubmit={generate} className="space-y-4">
        {mode === "edit" && !initial && <PictureInput value={picked} onChange={setPicked} />}
        <div className="flex flex-wrap gap-4 text-sm">
          <label className="space-y-1">
            <span className="block text-[var(--muted)]">Provider</span>
            <select
              value={providerId}
              onChange={(e) => setProviderId(e.target.value)}
              className="rounded border border-[var(--border)] bg-transparent px-2 py-1"
            >
              {usable.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.title}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-1">
            <span className="block text-[var(--muted)]">Base model</span>
            <select
              value={versionId}
              onChange={(e) => changeBase(e.target.value)}
              className="rounded border border-[var(--border)] bg-transparent px-2 py-1"
            >
              {bases.map((b) => (
                <option key={b.id} value={b.id}>
                  {b.name}
                </option>
              ))}
            </select>
          </label>
          {mode === "text" && (
            <label className="space-y-1">
              <span className="block text-[var(--muted)]">Size</span>
              <select
                value={size}
                onChange={(e) => setSize(e.target.value as typeof size)}
                className="rounded border border-[var(--border)] bg-transparent px-2 py-1"
              >
                {SIZES.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
        {lost.length > 0 && (
          <p
            role="alert"
            className="rounded border border-amber-600 px-3 py-2 text-sm text-amber-600"
          >
            Not available any more: {lost.map((l) => l.name).join(", ")}. The remix runs without{" "}
            {lost.length === 1 ? "it" : "them"}.
          </p>
        )}
        <fieldset className="space-y-2 text-sm">
          <legend className="text-[var(--muted)]">LoRAs</legend>
          {note && <p className="text-amber-600">{note}</p>}
          {stack.map((s) => {
            const l = loras.find((x) => x.id === s.versionId);
            if (!l) return null;
            return (
              <div
                key={s.versionId}
                className="space-y-1 rounded border border-[var(--border)] p-2"
              >
                <div className="flex items-center gap-3">
                  <span className="min-w-28 font-medium">{l.name}</span>
                  <input
                    type="range"
                    aria-label={`Weight of ${l.name}`}
                    min={-1}
                    max={2}
                    step={0.05}
                    value={s.weight}
                    onChange={(e) =>
                      setStack(
                        stack.map((x) =>
                          x.versionId === s.versionId
                            ? { ...x, weight: Number(e.target.value) }
                            : x,
                        ),
                      )
                    }
                    className="flex-1"
                  />
                  <span className="w-10 text-right tabular-nums">{s.weight.toFixed(2)}</span>
                  <button
                    type="button"
                    aria-label={`Remove ${l.name}`}
                    onClick={() => setStack(stack.filter((x) => x.versionId !== s.versionId))}
                    className="text-[var(--muted)]"
                  >
                    ✕
                  </button>
                </div>
                {l.triggerWords.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {l.triggerWords.map((w) => (
                      <button
                        type="button"
                        key={w}
                        onClick={() => addTrigger(w)}
                        className="rounded-full border border-[var(--border)] px-2 py-0.5 text-xs"
                      >
                        + {w}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
          {stack.length < 5 && (
            <select
              aria-label="Add a LoRA"
              value=""
              disabled={available.length === 0}
              onChange={(e) => {
                if (e.target.value) setStack([...stack, { versionId: e.target.value, weight: 1 }]);
              }}
              className="rounded border border-[var(--border)] bg-transparent px-2 py-1"
            >
              <option value="">
                {available.length > 0
                  ? "Add a LoRA…"
                  : loras.some((l) => fits(l, family))
                    ? "All compatible LoRAs are added"
                    : "No LoRA fits this base model"}
              </option>
              {available.map((l) => (
                <option key={l.id} value={l.id}>
                  {l.name}
                </option>
              ))}
            </select>
          )}
        </fieldset>
        {initial?.seed != null && (
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={keepSeed}
              onChange={(e) => setKeepSeed(e.target.checked)}
            />
            Keep the same seed ({initial.seed})
          </label>
        )}
        <label className="block space-y-1 text-sm">
          <span className="block text-[var(--muted)]">
            {mode === "edit" ? "What should change?" : "Prompt"}
          </span>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            maxLength={500}
            rows={4}
            placeholder={
              mode === "edit"
                ? "Make it dusk, with warm lights"
                : "A red fox in fresh snow at dawn, soft light"
            }
            className="w-full rounded border border-[var(--border)] bg-transparent px-3 py-2"
          />
          <span className="block text-right text-xs text-[var(--muted)]">{prompt.length}/500</span>
        </label>
        <button
          type="submit"
          disabled={working || !!blocked || !prompt.trim() || needsPicture}
          className="rounded bg-[var(--fg)] px-4 py-2 text-sm text-[var(--bg)] disabled:opacity-50"
        >
          {working ? "Working…" : "Generate"}
        </button>
        {working && (
          <button type="button" onClick={cancel} className="ml-3 text-sm underline">
            Cancel
          </button>
        )}
      </form>

      {phase.name === "working" && (
        <div className="space-y-1" aria-live="polite">
          <div className="flex justify-between text-sm text-[var(--muted)]">
            <span>{phase.progress >= 1 ? "Saving" : phase.label}</span>
            <span>{Math.round(phase.progress * 100)}%</span>
          </div>
          <div className="h-2 rounded bg-[var(--border)]">
            <div
              className="h-2 rounded bg-[var(--fg)] transition-all"
              style={{ width: `${Math.round(phase.progress * 100)}%` }}
            />
          </div>
        </div>
      )}
      {phase.name === "failed" && (
        <p role="alert" className="text-sm text-red-600">
          {phase.message}
        </p>
      )}
      {phase.name === "done" && (
        <div className="space-y-2">
          <Link href={`/creations/${phase.creationId}`}>
            <img
              src={`/api/creations/${phase.creationId}/image`}
              alt={prompt}
              className="max-h-[70vh] rounded border border-[var(--border)]"
            />
          </Link>
          <p className="text-sm text-[var(--muted)]">Saved to your creations.</p>
        </div>
      )}
    </div>
  );
}
