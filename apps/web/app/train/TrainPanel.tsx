"use client";

import type { DatasetSummary } from "@wasifyai-art/contracts";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

export function TrainPanel({ initial }: { initial: DatasetSummary[] }) {
  const router = useRouter();
  const [datasets, setDatasets] = useState(initial);
  const [name, setName] = useState("");
  const [trigger, setTrigger] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function create(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const r = await fetch("/api/datasets", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name: name.trim(), trigger_word: trigger.trim() }),
      });
      const body = await r.json();
      if (!r.ok)
        throw new Error(typeof body.detail === "string" ? body.detail : "Could not create");
      router.push(`/train/${(body as DatasetSummary).id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not create");
      setBusy(false);
    }
  }

  async function remove(id: string) {
    const r = await fetch(`/api/datasets/${id}`, { method: "DELETE" });
    if (r.ok) setDatasets((d) => d.filter((x) => x.id !== id));
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-xl font-medium">Train a LoRA</h1>
        <p className="text-sm text-[var(--muted)]">
          Upload a few pictures of a subject or style, train on this Mac, then publish the result
          into your models.
        </p>
      </header>

      <form onSubmit={create} className="flex flex-wrap items-end gap-3 text-sm">
        <label className="space-y-1">
          <span className="block text-[var(--muted)]">Dataset name</span>
          <input
            required
            maxLength={80}
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="rounded border border-[var(--border)] bg-transparent px-2 py-1"
          />
        </label>
        <label className="space-y-1">
          <span className="block text-[var(--muted)]">Trigger word</span>
          <input
            required
            minLength={2}
            maxLength={40}
            pattern="[A-Za-z0-9_\-]+"
            title="Letters, digits, - and _"
            value={trigger}
            onChange={(e) => setTrigger(e.target.value)}
            className="rounded border border-[var(--border)] bg-transparent px-2 py-1"
          />
        </label>
        <button
          type="submit"
          disabled={busy}
          className="rounded bg-[var(--fg)] px-3 py-1.5 text-[var(--bg)] disabled:opacity-60"
        >
          {busy ? "Creating…" : "Create dataset"}
        </button>
      </form>
      {error && (
        <p role="alert" className="text-sm text-red-600">
          {error}
        </p>
      )}

      {datasets.length === 0 ? (
        <p className="text-sm text-[var(--muted)]">No datasets yet.</p>
      ) : (
        <ul className="divide-y divide-[var(--border)] rounded border border-[var(--border)] text-sm">
          {datasets.map((d) => (
            <li key={d.id} className="flex items-center gap-4 px-4 py-3">
              <Link href={`/train/${d.id}`} className="min-w-0 flex-1">
                <span className="font-medium">{d.name}</span>
                <span className="ml-3 text-[var(--muted)]">
                  {d.trigger_word} · {d.image_count} pictures
                </span>
              </Link>
              <button
                type="button"
                onClick={() => remove(d.id)}
                className="text-[var(--muted)] hover:text-red-600"
              >
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
