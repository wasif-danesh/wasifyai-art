import { apiBase } from "@/lib/api";

const FILES = new Set(["image", "thumb"]);

export async function GET(
  _: Request,
  { params }: { params: Promise<{ id: string; file: string }> },
) {
  const { id, file } = await params;
  if (!FILES.has(file)) return new Response("not found", { status: 404 });
  const r = await fetch(`${apiBase()}/v1/creations/${encodeURIComponent(id)}/${file}`, {
    cache: "no-store",
  });
  if (!r.ok) return new Response("not found", { status: r.status });
  return new Response(r.body, {
    headers: {
      "content-type": r.headers.get("content-type") ?? "application/octet-stream",
      "cache-control": "private, max-age=3600",
    },
  });
}
