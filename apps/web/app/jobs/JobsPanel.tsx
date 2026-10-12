"use client";

import type { JobSummary } from "@wasifyai-art/contracts";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

const ACTIVE = new Set(["queued", "running"]);

export function JobsPanel() {
  const [jobs, setJobs] = useState<JobSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    const r = await fetch("/api/jobs", { cache: "no-store" });
    if (r.ok) setJobs(((await r.json()) as { jobs: JobSummary[] }).jobs);
    else setError("Could not load jobs.");
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, [load]);

  async function act(id: string, what: "cancel" | "retry") {
    setError(null);
    const r = await fetch(`/api/jobs/${id}/${what}`, { method: "POST" });
    if (!r.ok) setError(((await r.json()) as { detail?: string }).detail ?? "That did not work.");
    await load();
  }

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-medium">Jobs</h1>
      {error && (
        <p role="alert" className="text-sm text-red-600">
          {error}
        </p>
      )}
      {jobs === null ? (
        <p className="text-sm text-[var(--muted)]">Loading…</p>
      ) : jobs.length === 0 ? (
        <p className="text-sm text-[var(--muted)]">No jobs yet. Create an image to start.</p>
      ) : (
        <ul className="divide-y divide-[var(--border)] rounded border border-[var(--border)] text-sm">
          {jobs.map((j) => (
            <li key={j.id} className="flex items-center gap-4 px-3 py-2">
              <div className="min-w-0 flex-1">
                <p className="truncate">{j.prompt}</p>
                <p className="text-xs text-[var(--muted)]">
                  {j.model_name} · {j.mode === "edit" ? "picture" : "text"} ·{" "}
                  {new Date(j.created_at).toLocaleTimeString()}
                </p>
              </div>
              <span
                className={
                  j.status === "failed"
                    ? "text-red-600"
                    : j.status === "succeeded"
                      ? "text-green-600"
                      : "text-[var(--muted)]"
                }
              >
                {j.status}
                {j.status === "running" ? ` ${Math.round(j.progress * 100)}%` : ""}
              </span>
              {j.error && <span className="max-w-60 truncate text-xs text-red-600">{j.error}</span>}
              {j.creation_id && (
                <Link href={`/creations/${j.creation_id}`} className="underline">
                  Open
                </Link>
              )}
              {ACTIVE.has(j.status) && (
                <button type="button" onClick={() => act(j.id, "cancel")} className="underline">
                  Cancel
                </button>
              )}
              {(j.status === "failed" || j.status === "canceled") && (
                <button type="button" onClick={() => act(j.id, "retry")} className="underline">
                  Retry
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
