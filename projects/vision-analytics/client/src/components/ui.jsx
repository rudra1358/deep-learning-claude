export function PageHeader({ title, description, actions }) {
  return (
    <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {description && <p className="mt-1 max-w-3xl text-sm text-muted">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </div>
  );
}

export function Card({ title, actions, children, className = "" }) {
  return (
    <section className={`min-w-0 rounded-lg border border-border bg-surface p-4 shadow-sm ${className}`}>
      {(title || actions) && (
        <div className="mb-3 flex items-center justify-between gap-2">
          {title && <h2 className="text-sm font-semibold">{title}</h2>}
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

export function StatCard({ label, value, hint }) {
  return (
    <div className="min-w-0 rounded-lg border border-border bg-surface p-4 shadow-sm">
      <div className="text-xs font-medium tracking-wide text-muted uppercase">{label}</div>
      <div className="mt-1 truncate text-2xl font-semibold tabular-nums">{value}</div>
      {hint && <div className="mt-1 text-xs text-muted">{hint}</div>}
    </div>
  );
}

export function Spinner({ className = "h-4 w-4" }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeOpacity="0.25" strokeWidth="4" />
      <path d="M22 12a10 10 0 0 0-10-10" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
    </svg>
  );
}

export function LoadingState({ label = "Loading…" }) {
  return (
    <div role="status" className="flex items-center justify-center gap-2 py-10 text-sm text-muted">
      <Spinner /> {label}
    </div>
  );
}

export function ErrorState({ message, onRetry }) {
  return (
    <div role="alert" className="rounded-md border border-danger/40 bg-danger/10 p-3 text-sm text-danger">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="break-words">{message}</span>
        {onRetry && (
          <button type="button" className="btn" onClick={onRetry}>
            Retry
          </button>
        )}
      </div>
    </div>
  );
}

export function EmptyState({ children }) {
  return (
    <div className="rounded-md border border-dashed border-border p-6 text-center text-sm text-muted">{children}</div>
  );
}

const STATUS_STYLE = {
  done: "bg-success/15 text-success",
  running: "bg-accent/15 text-accent",
  queued: "bg-warning/15 text-warning",
  failed: "bg-danger/15 text-danger",
};

export function StatusBadge({ status }) {
  const style = STATUS_STYLE[status] ?? "bg-surface-2 text-muted";
  return <span className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${style}`}>{status}</span>;
}

export function KindBadge({ kind }) {
  return (
    <span className="inline-block rounded-full border border-border px-2 py-0.5 text-xs font-medium text-muted">
      {kind}
    </span>
  );
}

export function ProgressBar({ value, label }) {
  const pct = Math.round(Math.min(Math.max(value, 0), 1) * 100);
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs text-muted">
        <span>{label}</span>
        <span className="tabular-nums">{pct}%</span>
      </div>
      <div
        role="progressbar"
        aria-label={label}
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        className="h-2 overflow-hidden rounded-full bg-surface-2"
      >
        <div className="h-full rounded-full bg-accent transition-[width] duration-500" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function Slider({ id, label, value, min, max, step, onChange, disabled }) {
  return (
    <div>
      <label htmlFor={id} className="label flex justify-between">
        <span>{label}</span>
        <span className="tabular-nums normal-case">{value.toFixed(2)}</span>
      </label>
      <input
        id={id}
        type="range"
        className="w-full"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

/** Two-column definition list for metadata. */
export function KeyValue({ items }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
      {items.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-muted">{k}</dt>
          <dd className="min-w-0 break-words tabular-nums">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

/** Timing breakdown as a stacked bar + numbers (schemas.Timing). */
export function TimingBreakdown({ timing }) {
  const parts = [
    { k: "Preprocess", v: timing.preprocess_ms, c: "bg-warning" },
    { k: "Inference", v: timing.inference_ms, c: "bg-accent" },
    { k: "Postprocess", v: timing.postprocess_ms, c: "bg-success" },
  ];
  const total = Math.max(timing.total_ms, 1e-6);
  return (
    <div>
      <div className="flex h-3 overflow-hidden rounded-full bg-surface-2" aria-hidden>
        {parts.map((p) => (
          <div key={p.k} className={p.c} style={{ width: `${(p.v / total) * 100}%` }} />
        ))}
      </div>
      <ul className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        {parts.map((p) => (
          <li key={p.k} className="flex items-center gap-1.5">
            <span className={`h-2 w-2 rounded-sm ${p.c}`} aria-hidden />
            <span className="text-muted">{p.k}</span>
            <span className="ml-auto whitespace-nowrap tabular-nums">{p.v.toFixed(1)} ms</span>
          </li>
        ))}
        <li className="flex items-center gap-1.5 font-medium">
          <span className="text-muted">Total</span>
          <span className="ml-auto whitespace-nowrap tabular-nums">{timing.total_ms.toFixed(1)} ms</span>
        </li>
      </ul>
    </div>
  );
}
