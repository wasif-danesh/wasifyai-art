"use client";

import type { DatasetSummary, TrainingPreset, TrainingSummary } from "@wasifyai-art/contracts";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

// API paths come back as /v1/...; the browser reaches them through the BFF under /api/.
const bff = (url: string) => url.replace(/^\/v1\//, "/api/");

const PRESETS: { id: TrainingPreset; label: string }[] = [
  { id: "fast", label: "Fast (quick test)" },
  { id: "balanced", label: "Balanced" },
  { id: "quality", label: "Quality (longest)" },
];

async function detail(r: Response, fallback: string): Promise<string> {
  try {
    const b = (await r.json()) as { detail?: unknown };
    return typeof b.detail === "string" ? b.detail : fallback;
  } catch {
    return fallback;
  }
}

export function DatasetEditor({ initial }: { initial: DatasetSummary }) {
  const [dataset, setDataset] = useState(initial);
  const [runs, setRuns] = useState<TrainingSummary[]>([]);
  const [preset, setPreset] = useState<TrainingPreset>("fast");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [published, setPublished] = useState<string | null>(null);

  const reload = useCallback(async () => {
    const [d, t] = await Promise.all([
      fetch(`/api/datasets/${initial.id}`, { cache: "no-store" }),
      fetch(`/api/trainings?dataset_id=${initial.id}`, { cache: "no-store" }),
    ]);
    if (d.ok) setDataset((await d.json()) as DatasetSummary);
    if (t.ok) setRuns(((await t.json()) as { trainings: TrainingSummary[] }).trainings);
  }, [initial.id]);

  useEffect(() => {
    reload();
    const t = setInterval(reload, 2500);
    return () => clearInterval(t);
  }, [reload]);

  async function upload(files: FileList | null) {
    if (!files?.length) return;
    setError(null);
    for (const [i, file] of Array.from(files).entries()) {
      setBusy(`Uploading ${i + 1} of ${files.length}…`);
      const r = await fetch(`/api/datasets/${dataset.id}/images`, {
        method: "POST",
        headers: { "content-type": file.type || "application/octet-stream" },
        body: file,
      });
      if (!r.ok) {
        setError(`${file.name}: ${await detail(r, "could not be added")}`);
        break;
      }
    }
    setBusy(null);
    await reload();
  }

  async function saveCaption(id: string, caption: string) {
    const r = await fetch(`/api/datasets/${dataset.id}/images/${id}`, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ caption }),
    });
    if (!r.ok) setError(await detail(r, "Could not save the caption"));
    await reload();
  }

  async function removeImage(id: string) {
    await fetch(`/api/datasets/${dataset.id}/images/${id}`, { method: "DELETE" });
    await reload();
  }

  async function start() {
    setError(null);
    setBusy("Starting…");
    const r = await fetch("/api/trainings", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ dataset_id: dataset.id, preset }),
    });
    if (!r.ok) setError(await detail(r, "Could not start training"));
    setBusy(null);
    await reload();
  }

  async function cancel(jobId: string) {
    await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
    await reload();
  }

  async function publish(run: TrainingSummary, step: number) {
    setError(null);
    setPublished(null);
    setBusy(`Publishing step ${step}…`);
    const r = await fetch(`/api/trainings/${run.id}/publish`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ step }),
    });
    setBusy(null);
    if (!r.ok) {
      setError(await detail(r, "Could not publish"));
      return;
    }
    setPublished(run.trigger_word);
  }

  const active = runs.some((r) => r.status === "queued" || r.status === "running");

  return (
    <div className="space-y-8">
      <header>
        <Link href="/train" className="text-sm text-[var(--muted)]">
          ← Datasets
        </Link>
        <h1 className="text-xl font-medium">{dataset.name}</h1>
        <p className="text-sm text-[var(--muted)]">
          Trigger word <code>{dataset.trigger_word}</code> · {dataset.image_count} pictures. Each
          caption should describe the picture and include the trigger word.
        </p>
      </header>

      {error && (
        <p role="alert" className="rounded border border-red-600 px-3 py-2 text-sm text-red-600">
          {error}
        </p>
      )}
      {published && (
        <output className="block text-sm text-green-600">
          Published. Find it in{" "}
          <Link href="/models" className="underline">
            Models
          </Link>
          , then use “{published}” in a prompt on Create.
        </output>
      )}

      <section className="space-y-3">
        <label className="inline-block cursor-pointer rounded border border-[var(--border)] px-3 py-1.5 text-sm">
          Add pictures
          <input
            type="file"
            accept="image/png,image/jpeg,image/webp"
            multiple
            className="sr-only"
            onChange={(e) => {
              upload(e.target.files);
              e.target.value = "";
            }}
          />
        </label>
        {busy && <span className="ml-3 text-sm text-[var(--muted)]">{busy}</span>}

        <ul className="grid gap-3 sm:grid-cols-2 md:grid-cols-3">
          {dataset.images.map((img) => (
            <li key={img.id} className="space-y-2 rounded border border-[var(--border)] p-2">
              <img
                src={bff(img.thumb_url)}
                alt={img.caption}
                width={img.width}
                height={img.height}
                className="h-40 w-full rounded object-cover"
              />
              <CaptionField
                key={img.caption}
                value={img.caption}
                warn={!img.has_trigger}
                onSave={(c) => saveCaption(img.id, c)}
              />
              <button
                type="button"
                onClick={() => removeImage(img.id)}
                className="text-xs text-[var(--muted)] hover:text-red-600"
              >
                Remove
              </button>
            </li>
          ))}
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="font-medium">Train</h2>
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <label className="sr-only" htmlFor="preset">
            Preset
          </label>
          <select
            id="preset"
            value={preset}
            onChange={(e) => setPreset(e.target.value as TrainingPreset)}
            className="rounded border border-[var(--border)] bg-transparent px-2 py-1"
          >
            {PRESETS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={start}
            disabled={active || dataset.image_count < 5 || busy !== null}
            className="rounded bg-[var(--fg)] px-3 py-1.5 text-[var(--bg)] disabled:opacity-60"
          >
            Start training
          </button>
          {dataset.image_count < 5 && (
            <span className="text-[var(--muted)]">Add at least 5 pictures.</span>
          )}
        </div>

        {runs.length > 0 && (
          <ul className="space-y-3">
            {runs.map((run) => (
              <li
                key={run.id}
                className="space-y-2 rounded border border-[var(--border)] p-3 text-sm"
              >
                <div className="flex items-center gap-3">
                  <span
                    className={
                      run.status === "failed"
                        ? "text-red-600"
                        : run.status === "succeeded"
                          ? "text-green-600"
                          : ""
                    }
                  >
                    {run.status}
                  </span>
                  <span className="text-[var(--muted)]">
                    {run.preset} · {run.steps} steps · {new Date(run.created_at).toLocaleString()}
                  </span>
                  {(run.status === "queued" || run.status === "running") && (
                    <>
                      <progress value={run.progress} max={1} className="flex-1" />
                      <button type="button" onClick={() => cancel(run.id)} className="underline">
                        Cancel
                      </button>
                    </>
                  )}
                </div>
                {run.error && <p className="text-red-600">{run.error}</p>}
                {run.checkpoints.length > 0 && (
                  <ul className="grid gap-3 sm:grid-cols-3 md:grid-cols-4">
                    {run.checkpoints.map((cp) => (
                      <li key={cp.step} className="space-y-1">
                        {cp.preview_url && (
                          <img
                            src={bff(cp.preview_url)}
                            alt={`Preview at step ${cp.step}`}
                            className="aspect-square w-full rounded object-cover"
                          />
                        )}
                        <div className="flex items-center justify-between">
                          <span>Step {cp.step}</span>
                          <button
                            type="button"
                            disabled={busy !== null}
                            onClick={() => publish(run, cp.step)}
                            className="underline disabled:opacity-60"
                          >
                            Publish
                          </button>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function CaptionField({
  value,
  warn,
  onSave,
}: {
  value: string;
  warn: boolean;
  onSave: (caption: string) => void;
}) {
  const [text, setText] = useState(value);
  return (
    <div>
      <textarea
        aria-label="Caption"
        rows={2}
        maxLength={1000}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={() => text !== value && onSave(text)}
        className="w-full rounded border border-[var(--border)] bg-transparent px-2 py-1 text-xs"
      />
      {warn && <p className="text-xs text-amber-600">Caption is missing the trigger word.</p>}
    </div>
  );
}
