import { apiBase } from "@/lib/api";

export async function GET() {
  const r = await fetch(`${apiBase()}/v1/jobs`, { cache: "no-store" });
  return new Response(await r.text(), {
    status: r.status,
    headers: { "content-type": "application/json" },
  });
}
