import { apiBase } from "@/lib/api";

export async function POST(_: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const r = await fetch(`${apiBase()}/v1/jobs/${encodeURIComponent(id)}/retry`, {
    method: "POST",
    cache: "no-store",
  });
  return new Response(await r.text(), {
    status: r.status,
    headers: { "content-type": "application/json" },
  });
}
