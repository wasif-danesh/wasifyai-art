import { apiBase } from "@/lib/api";

export async function POST(req: Request) {
  const r = await fetch(`${apiBase()}/v1/generations`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: await req.text(),
    cache: "no-store",
  });
  return new Response(await r.text(), {
    status: r.status,
    headers: { "content-type": "application/json" },
  });
}
