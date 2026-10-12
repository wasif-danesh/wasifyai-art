import { apiBase } from "@/lib/api";

export async function DELETE(_: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const r = await fetch(`${apiBase()}/v1/uploads/${encodeURIComponent(id)}`, {
    method: "DELETE",
    cache: "no-store",
  });
  return new Response(null, { status: r.status });
}
