"use client";

import { useRef, useState } from "react";

const TYPES = ["image/png", "image/jpeg", "image/webp"];
const MAX_BYTES = 10 * 1024 * 1024;

export type Picked = { uploadId: string; previewUrl: string; width: number; height: number };

/** Choose, drop or paste one picture. It is uploaded at once and can be removed (which deletes it). */
export function PictureInput({
  value,
  onChange,
}: {
  value: Picked | null;
  onChange: (p: Picked | null) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [over, setOver] = useState(false);

  async function take(file: File | undefined) {
    if (!file) return;
    setError(null);
    if (!TYPES.includes(file.type)) return setError("Use a PNG, JPEG or WebP picture.");
    if (file.size > MAX_BYTES) return setError("That picture is too large (over 10 MB).");
    setBusy(true);
    try {
      if (value) await remove();
      const r = await fetch("/api/uploads/images", {
        method: "POST",
        headers: { "content-type": file.type },
        body: file,
      });
      const body = await r.json();
      if (!r.ok) return setError(typeof body.detail === "string" ? body.detail : "Upload failed");
      onChange({
        uploadId: body.upload_id,
        previewUrl: URL.createObjectURL(file),
        width: body.width,
        height: body.height,
      });
    } catch {
      setError("Upload failed. Check your connection and try again.");
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    if (value) {
      await fetch(`/api/uploads/${value.uploadId}`, { method: "DELETE" });
      URL.revokeObjectURL(value.previewUrl);
      onChange(null);
    }
  }

  return (
    <div
      onPaste={(e) =>
        take(Array.from(e.clipboardData.files).find((f) => f.type.startsWith("image/")))
      }
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        take(e.dataTransfer.files[0]);
      }}
      className={`rounded border border-dashed p-4 text-sm ${over ? "border-[var(--fg)]" : "border-[var(--border)]"}`}
    >
      {value ? (
        <div className="flex items-center gap-4">
          <img
            src={value.previewUrl}
            alt="Preview of your upload"
            className="h-24 rounded border border-[var(--border)]"
          />
          <div className="space-y-1">
            <p className="text-[var(--muted)]">
              {value.width}×{value.height}. It is deleted after the image is made.
            </p>
            <button type="button" onClick={remove} className="underline">
              Remove
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-2 text-center text-[var(--muted)]">
          <p>{busy ? "Uploading…" : "Drop a picture here, paste one, or"}</p>
          <button
            type="button"
            disabled={busy}
            onClick={() => input.current?.click()}
            className="rounded border border-[var(--border)] px-3 py-1 text-[var(--fg)]"
          >
            Choose a picture
          </button>
          <input
            ref={input}
            type="file"
            accept={TYPES.join(",")}
            className="sr-only"
            aria-label="Choose a picture"
            onChange={(e) => take(e.target.files?.[0])}
          />
        </div>
      )}
      {error && (
        <p role="alert" className="mt-2 text-red-600">
          {error}
        </p>
      )}
    </div>
  );
}
