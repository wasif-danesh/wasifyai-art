import { apiBase } from "@/lib/api";

// BFF: the browser calls this route, never the API directly.
export async function POST() {
  const r = await fetch(`${apiBase()}/v1/models/scan`, { method: "POST", cache: "no-store" });
  return new Response(await r.text(), {
    status: r.status,
    headers: { "content-type": "application/json" },
  });
}
