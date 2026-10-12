import { apiBase } from "@/lib/api";

const MAX = 10 * 1024 * 1024;

// The picture is forwarded as the raw body. The API enforces the real limits; this only avoids
// buffering something absurd.
export async function POST(req: Request) {
  const declared = Number(req.headers.get("content-length") ?? "0");
  if (declared > MAX) {
    return Response.json({ detail: "That picture is too large (over 10 MB)." }, { status: 413 });
  }
  const r = await fetch(`${apiBase()}/v1/uploads/images`, {
    method: "POST",
    headers: { "content-type": req.headers.get("content-type") ?? "application/octet-stream" },
    body: await req.arrayBuffer(),
    cache: "no-store",
  });
  return new Response(await r.text(), {
    status: r.status,
    headers: { "content-type": "application/json" },
  });
}
