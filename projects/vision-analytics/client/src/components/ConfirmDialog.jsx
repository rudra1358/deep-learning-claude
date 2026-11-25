import { useEffect, useRef } from "react";

/** Accessible modal confirmation built on the native <dialog> element (focus trap + Esc for free). */
export function ConfirmDialog({ open, title, message, confirmLabel = "Confirm", busy, onConfirm, onCancel }) {
  const ref = useRef(null);

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      aria-labelledby="confirm-title"
      onCancel={(e) => {
        e.preventDefault();
        onCancel();
      }}
      className="m-auto w-[min(28rem,calc(100vw-2rem))] rounded-lg border border-border bg-surface p-5 text-fg shadow-xl"
    >
      <h2 id="confirm-title" className="text-lg font-semibold">
        {title}
      </h2>
      <div className="mt-2 text-sm text-muted">{message}</div>
      <div className="mt-5 flex justify-end gap-2">
        <button type="button" className="btn" onClick={onCancel} disabled={busy} autoFocus>
          Cancel
        </button>
        <button type="button" className="btn btn-danger" onClick={onConfirm} disabled={busy}>
          {busy ? "Working…" : confirmLabel}
        </button>
      </div>
    </dialog>
  );
}
