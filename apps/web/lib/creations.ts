import type { CreationSummary } from "@wasifyai-art/contracts";

/** The browser-facing address of a creation's file (the API route is reached through the BFF). */
export function fileUrl(c: CreationSummary, which: "image" | "thumb"): string {
  return `/api/creations/${c.id}/${which}`;
}
