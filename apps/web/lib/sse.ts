export type JobEvent =
  | { type: "queued" }
  | { type: "running" }
  | { type: "progress"; progress: number }
  | { type: "succeeded"; creationId: string }
  | { type: "failed"; message: string }
  | { type: "canceled" };

/** Parse one SSE data payload for an event type; unknown types return null. */
export function parseJobEvent(type: string, data: string): JobEvent | null {
  let body: Record<string, unknown> = {};
  try {
    body = JSON.parse(data) as Record<string, unknown>;
  } catch {
    return null;
  }
  switch (type) {
    case "queued":
    case "running":
    case "canceled":
      return { type };
    case "progress":
      return typeof body.progress === "number" ? { type, progress: body.progress } : null;
    case "succeeded":
      return typeof body.creation_id === "string" ? { type, creationId: body.creation_id } : null;
    case "failed":
      return { type, message: typeof body.message === "string" ? body.message : "Failed" };
    default:
      return null;
  }
}
