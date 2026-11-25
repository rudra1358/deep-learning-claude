"use client";

import { useRef, useState } from "react";
import { errorMessage, uploadDocuments } from "@/lib/api";

const ACCEPT = [".pdf", ".docx", ".txt", ".md"]; // the types backend/ingest/extract.py supports

// Drag-and-drop area (or click to pick files). Uploads with one multipart POST, then tells the
// parent to reload the table. The server answers 202 right away; processing happens in the background.
export default function UploadZone({ onUploaded }: { onUploaded: () => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);

  async function upload(list: FileList | null) {
    const files = Array.from(list ?? []);
    if (files.length === 0) return;
    // Check the extension here too: a friendly message is better than a 415 from the server.
    const bad = files.filter((f) => !ACCEPT.some((ext) => f.name.toLowerCase().endsWith(ext)));
    if (bad.length) {
      setMessage({ ok: false, text: `Unsupported type: ${bad.map((f) => f.name).join(", ")}. Use ${ACCEPT.join(" ")}.` });
      return;
    }
    setUploading(true);
    setMessage(null);
    try {
      const created = await uploadDocuments(files);
      setMessage({ ok: true, text: `Uploaded ${created.length} file(s). Processing in the background…` });
      onUploaded();
    } catch (e) {
      setMessage({ ok: false, text: errorMessage(e) });
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = ""; // allow picking the same file again
    }
  }

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload documents: drop files here or press Enter to choose"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
        // preventDefault on dragover is what tells the browser "dropping here is allowed"
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault(); // stop the browser from opening the file itself
          setDragging(false);
          upload(e.dataTransfer.files);
        }}
        className={`cursor-pointer rounded border-2 border-dashed p-8 text-center text-sm ${
          dragging ? "border-accent bg-accent/10" : "border-border bg-card"
        }`}
      >
        <p className="font-medium">{uploading ? "Uploading…" : "Drop files here or click to choose"}</p>
        <p className="mt-1 text-muted">PDF, DOCX, TXT or Markdown · several at once is fine</p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPT.join(",")}
          onChange={(e) => upload(e.target.files)}
          className="hidden"
        />
      </div>
      {message && (
        <p role="status" className={`mt-2 text-sm ${message.ok ? "text-ok" : "text-bad"}`}>
          {message.text}
        </p>
      )}
    </div>
  );
}
