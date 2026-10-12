import { apiBase } from "@/lib/api";

export const dynamic = "force-dynamic";

// Server-sent events are passed through untouched; the browser never talks to the API directly.
export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const headers: Record<string, string> = {};
  const last = req.headers.get("last-event-id");
  if (last) headers["last-event-id"] = last;
  const r = await fetch(`${apiBase()}/v1/jobs/${encodeURIComponent(id)}/events`, {
    headers,
    cache: "no-store",
    signal: req.signal,
  });
  if (!r.ok || !r.body) return new Response(await r.text(), { status: r.status });
  return new Response(r.body, {
    headers: { "content-type": "text/event-stream", "cache-control": "no-cache" },
  });
}
