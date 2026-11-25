// Tiny shared building blocks for the three states every data view has: loading, empty and error.
// Keeping them in one place makes every page look and behave the same.

export function Loading({ text = "Loading…" }: { text?: string }) {
  return (
    <p role="status" className="py-4 text-sm text-muted">
      {text}
    </p>
  );
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <p className="rounded border border-dashed border-border p-6 text-center text-sm text-muted">{children}</p>;
}

export function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex flex-wrap items-center gap-3 rounded border border-bad/40 bg-bad/10 p-3 text-sm text-bad">
      <span className="min-w-0 flex-1 break-words">{message}</span>
      {onRetry && (
        <button type="button" onClick={onRetry} className="rounded border border-bad/40 px-2 py-1">
          Retry
        </button>
      )}
    </div>
  );
}

// Coloured pill for document status / enrichment status.
const BADGE_COLORS: Record<string, string> = {
  ready: "text-ok border-ok/40",
  done: "text-ok border-ok/40",
  processing: "text-warn border-warn/40",
  running: "text-warn border-warn/40",
  queued: "text-muted border-border",
  pending: "text-muted border-border",
  failed: "text-bad border-bad/40",
};

export function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`inline-block rounded-full border px-2 py-0.5 text-xs ${BADGE_COLORS[status] ?? "border-border"}`}>
      {status}
    </span>
  );
}
