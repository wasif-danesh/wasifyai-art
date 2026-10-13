import { apiBase } from "@/lib/api";

const MAX_BODY = 12 * 1024 * 1024;

/** Forward a request to the API under a fixed prefix, keeping method, body and content type.
 * Bytes pass through untouched so thumbnails and previews work. */
export async function forward(req: Request, prefix: string, path: string[] | undefined) {
  const tail = (path ?? []).map(encodeURIComponent).join("/");
  const search = new URL(req.url).search;
  const url = `${apiBase()}/v1/${prefix}${tail ? `/${tail}` : ""}${search}`;
  const hasBody = !["GET", "HEAD", "DELETE"].includes(req.method);
  if (hasBody && Number(req.headers.get("content-length") ?? "0") > MAX_BODY) {
    return Response.json({ detail: "That upload is too large." }, { status: 413 });
  }
  const r = await fetch(url, {
    method: req.method,
    headers: { "content-type": req.headers.get("content-type") ?? "application/json" },
    body: hasBody ? await req.arrayBuffer() : undefined,
    cache: "no-store",
  });
  const headers = new Headers();
  const type = r.headers.get("content-type");
  if (type) headers.set("content-type", type);
  const cache = r.headers.get("cache-control");
  if (cache) headers.set("cache-control", cache);
  return new Response(r.status === 204 ? null : r.body, { status: r.status, headers });
}

export type Ctx = { params: Promise<{ path?: string[] }> };
