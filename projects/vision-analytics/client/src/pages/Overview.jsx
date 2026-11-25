import { Link } from "react-router-dom";
import { getHealth, getOverview, listRuns } from "../lib/api";
import { useAsync, usePageTitle } from "../lib/hooks";
import { fmtMs, sortedEntries, sumValues } from "../lib/format";
import { BarList } from "../components/charts";
import { RunsTable } from "../components/RunsTable";
import { Card, EmptyState, ErrorState, LoadingState, PageHeader, StatCard } from "../components/ui";

/**
 * Overview: three independent requests so one failing (e.g. no runs yet) never blanks the page.
 *   GET /api/analytics/overview  -> totals
 *   GET /api/health              -> providers + models
 *   GET /api/runs?limit=8        -> recent runs
 */
export default function OverviewPage() {
  usePageTitle();
  const overview = useAsync((signal) => getOverview({ signal }), []);
  const health = useAsync((signal) => getHealth({ signal }), []);
  const recent = useAsync((signal) => listRuns({ limit: 8 }, { signal }), []);

  const o = overview.data;
  const totalRuns = sumValues(o?.runs_by_kind);
  const crossIn = sumValues(o?.crossings.in);
  const crossOut = sumValues(o?.crossings.out);
  // placeholder for stat cards while loading ("…") or when the request failed ("—")
  const ph = !o ? (overview.loading ? "…" : "—") : null;

  return (
    <>
      <PageHeader
        title="Overview"
        description="Totals across every image, video and live-camera run stored by the API."
        actions={
          <button
            type="button"
            className="btn"
            onClick={() => {
              overview.reload();
              health.reload();
              recent.reload();
            }}
          >
            Refresh
          </button>
        }
      />

      {overview.error && (
        <div className="mb-4">
          <ErrorState message={overview.error} onRetry={overview.reload} />
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard
          label="Runs"
          value={ph ?? totalRuns}
          hint={
            o && totalRuns > 0
              ? sortedEntries(o.runs_by_kind)
                  .map(([k, v]) => `${v} ${k}`)
                  .join(" · ")
              : o
                ? "no runs yet"
                : undefined
          }
        />
        <StatCard label="Objects" value={ph ?? sumValues(o?.objects_by_label)} hint="detections + unique tracks" />
        <StatCard
          label="Line crossings"
          value={ph ?? `${crossIn} in / ${crossOut} out`}
          hint="video + live runs with a counting line"
        />
        <StatCard label="Avg latency" value={ph ?? fmtMs(o?.avg_latency_ms)} hint="per frame / image, finished runs" />
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card title="Objects by label" className="lg:col-span-2">
          {overview.loading && !o ? (
            <LoadingState />
          ) : (
            <BarList
              data={sortedEntries(o?.objects_by_label).slice(0, 15)}
              emptyText="Run a detection to see label counts."
            />
          )}
        </Card>

        <Card title="System health">
          {health.loading && !health.data ? (
            <LoadingState label="Checking API…" />
          ) : health.error ? (
            <ErrorState message={health.error} onRetry={health.reload} />
          ) : health.data ? (
            <div className="space-y-3 text-sm">
              <div className="flex items-center gap-2">
                <span
                  className={`h-2.5 w-2.5 rounded-full ${health.data.status === "ok" ? "bg-success" : "bg-warning"}`}
                  aria-hidden
                />
                <span>
                  API status: <strong>{health.data.status}</strong>
                </span>
              </div>
              <div>
                <div className="label">Execution providers</div>
                <div className="flex flex-wrap gap-1">
                  {health.data.providers.map((p) => (
                    <code key={p} className="rounded bg-surface-2 px-1.5 py-0.5 text-xs">
                      {p}
                    </code>
                  ))}
                </div>
              </div>
              <div>
                <div className="label">Models ({health.data.models.length})</div>
                {health.data.models.length === 0 ? (
                  <p className="text-muted">No .onnx models found yet.</p>
                ) : (
                  <ul className="space-y-0.5">
                    {health.data.models.map((m) => (
                      <li key={m} className="flex items-center justify-between gap-2">
                        <code className="truncate text-xs">{m}</code>
                        {m === health.data?.default_model && <span className="text-xs text-muted">default</span>}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          ) : null}
        </Card>

        <Card title="Line crossings by label" className="lg:col-span-1">
          {o && (crossIn > 0 || crossOut > 0) ? (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-1">
              <div>
                <div className="label">In</div>
                <BarList data={sortedEntries(o.crossings.in)} />
              </div>
              <div>
                <div className="label">Out</div>
                <BarList data={sortedEntries(o.crossings.out)} />
              </div>
            </div>
          ) : (
            <EmptyState>No crossings yet — draw a counting line on the Video or Live page.</EmptyState>
          )}
        </Card>

        <Card
          title="Recent runs"
          className="lg:col-span-2"
          actions={
            <Link to="/history" className="text-sm text-accent hover:underline">
              View all
            </Link>
          }
        >
          {recent.loading && !recent.data ? (
            <LoadingState />
          ) : recent.error ? (
            <ErrorState message={recent.error} onRetry={recent.reload} />
          ) : recent.data && recent.data.items.length > 0 ? (
            <RunsTable runs={recent.data.items} compact />
          ) : (
            <EmptyState>
              No runs yet. Try{" "}
              <Link to="/image" className="text-accent underline">
                image detection
              </Link>{" "}
              or{" "}
              <Link to="/live" className="text-accent underline">
                the live camera
              </Link>
              .
            </EmptyState>
          )}
        </Card>
      </div>
    </>
  );
}
