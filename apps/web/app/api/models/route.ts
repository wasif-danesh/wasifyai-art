import { apiBase } from "@/lib/api";

export async function GET() {
  const r = await fetch(`${apiBase()}/v1/models`, { cache: "no-store" });
  return new Response(await r.text(), {
    status: r.status,
    headers: { "content-type": "application/json" },
  });
}
