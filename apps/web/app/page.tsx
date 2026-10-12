import { listCreations, listModels, listProviders } from "@/lib/api";
import { fileUrl } from "@/lib/creations";
import { providerHint } from "@/lib/labels";
import Link from "next/link";

export const dynamic = "force-dynamic";

export default async function Home() {
  const [providers, models, creations] = await Promise.all([
    listProviders(),
    listModels(),
    listCreations(),
  ]);
  return (
    <div className="space-y-8">
      <section className="flex items-end justify-between">
        <div>
          <h1 className="text-xl font-medium">Your creations</h1>
          <p className="text-sm text-[var(--muted)]">
            {creations.length === 0
              ? "Nothing here yet. Create your first image."
              : `${creations.length} ${creations.length === 1 ? "image" : "images"}`}
          </p>
        </div>
        <Link
          href="/create"
          className="rounded bg-[var(--fg)] px-3 py-1.5 text-sm text-[var(--bg)]"
        >
          Create
        </Link>
      </section>
      {creations.length > 0 && (
        <ul className="grid grid-cols-2 gap-3 md:grid-cols-4">
          {creations.map((c) => (
            <li key={c.id}>
              <Link href={`/creations/${c.id}`}>
                <img
                  src={fileUrl(c, "thumb")}
                  alt={c.prompt}
                  loading="lazy"
                  className="aspect-square w-full rounded border border-[var(--border)] object-cover"
                />
              </Link>
            </li>
          ))}
        </ul>
      )}
      <section>
        <h2 className="mb-2 text-sm font-medium text-[var(--muted)]">Providers</h2>
        <ul className="divide-y divide-[var(--border)] rounded border border-[var(--border)] text-sm">
          {providers.map((p) => (
            <li key={p.id} className="flex justify-between px-3 py-2">
              <span>{p.title}</span>
              <span className="text-[var(--muted)]">{providerHint(p) || p.kind}</span>
            </li>
          ))}
        </ul>
      </section>
      <section>
        <h2 className="mb-2 text-sm font-medium text-[var(--muted)]">Models ({models.length})</h2>
        {models.length === 0 && (
          <p className="text-sm text-[var(--muted)]">
            The hub is empty. Scan a model folder or upload a LoRA to start.
          </p>
        )}
      </section>
    </div>
  );
}
