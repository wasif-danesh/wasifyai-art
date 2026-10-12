import { getCreation } from "@/lib/api";
import { fileUrl } from "@/lib/creations";
import Link from "next/link";
import { notFound } from "next/navigation";

export const dynamic = "force-dynamic";

export default async function CreationPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const c = await getCreation(id);
  if (!c) notFound();
  return (
    <div className="grid gap-6 md:grid-cols-[minmax(0,1fr)_320px]">
      <img
        src={fileUrl(c, "image")}
        alt={c.prompt}
        className="w-full rounded border border-[var(--border)]"
      />
      <aside className="space-y-4 text-sm">
        <Link href="/" className="text-[var(--muted)]">
          ← Your creations
        </Link>
        <div className="flex gap-2">
          <Link
            href={`/create?remix=${c.id}`}
            className="rounded bg-[var(--fg)] px-3 py-1.5 text-[var(--bg)]"
          >
            Remix settings
          </Link>
          <Link
            href={`/create?edit=${c.id}`}
            className="rounded border border-[var(--border)] px-3 py-1.5"
          >
            Remix picture
          </Link>
        </div>
        {c.parent_id && (
          <p className="text-[var(--muted)]">
            Remixed from{" "}
            <Link href={`/creations/${c.parent_id}`} className="underline">
              another creation
            </Link>
            {c.params.mode === "edit" ? " (picture edit)" : ""}
          </p>
        )}
        <section>
          <h2 className="mb-1 text-xs uppercase tracking-wide text-[var(--muted)]">Prompt</h2>
          <p>{c.prompt}</p>
        </section>
        <section>
          <h2 className="mb-1 text-xs uppercase tracking-wide text-[var(--muted)]">Models used</h2>
          <ul>
            {c.models.map((m) => (
              <li key={m.rel_path}>
                {m.name}{" "}
                <span className="text-[var(--muted)]">
                  {m.role === "lora" ? `LoRA · ${m.weight ?? 1}` : (m.family ?? "")}
                </span>
              </li>
            ))}
          </ul>
        </section>
        <section>
          <h2 className="mb-1 text-xs uppercase tracking-wide text-[var(--muted)]">Settings</h2>
          <dl className="grid grid-cols-2 gap-y-1">
            <dt className="text-[var(--muted)]">Size</dt>
            <dd>
              {c.width}×{c.height}
            </dd>
            <dt className="text-[var(--muted)]">Seed</dt>
            <dd>{String(c.params.seed)}</dd>
            <dt className="text-[var(--muted)]">Steps</dt>
            <dd>{String(c.params.steps)}</dd>
            <dt className="text-[var(--muted)]">Provider</dt>
            <dd>{c.provider_id}</dd>
          </dl>
        </section>
      </aside>
    </div>
  );
}
