import { useId, useRef, useState } from "react";

/** Drag-and-drop zone that is also a keyboard-accessible file picker. */
export function FileDrop({ accept, onFile, label, hint, disabled, compact }) {
  const inputRef = useRef(null);
  const [over, setOver] = useState(false);
  const [err, setErr] = useState(null);
  const hintId = useId();

  const accepts = (f) =>
    accept.split(",").some((a) => {
      const t = a.trim();
      if (t.endsWith("/*")) return f.type.startsWith(t.slice(0, -1));
      if (t.startsWith(".")) return f.name.toLowerCase().endsWith(t.toLowerCase());
      return f.type === t;
    });

  const take = (f) => {
    if (!f) return;
    if (!accepts(f)) {
      setErr(`Unsupported file "${f.name}". Expected ${accept}.`);
      return;
    }
    setErr(null);
    onFile(f);
  };

  return (
    <div>
      <button
        type="button"
        disabled={disabled}
        aria-describedby={hintId}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          if (!disabled) take(e.dataTransfer.files[0]);
        }}
        className={`flex w-full flex-col items-center justify-center gap-1 rounded-lg border-2 border-dashed text-center transition-colors disabled:opacity-50 ${compact ? "p-4" : "p-8"} ${over ? "border-accent bg-accent/5" : "border-border hover:border-accent/60"}`}
      >
        <svg
          aria-hidden
          width="28"
          height="28"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          className="text-muted"
        >
          <path d="M12 16V4m0 0l-4 4m4-4l4 4M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" />
        </svg>
        <span className="text-sm font-medium">{label}</span>
        <span id={hintId} className="text-xs text-muted">
          {hint ?? "Drag & drop or click to choose"}
        </span>
      </button>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="sr-only"
        tabIndex={-1}
        aria-hidden
        onChange={(e) => {
          take(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      {err && (
        <p role="alert" className="mt-2 text-xs text-danger">
          {err}
        </p>
      )}
    </div>
  );
}
