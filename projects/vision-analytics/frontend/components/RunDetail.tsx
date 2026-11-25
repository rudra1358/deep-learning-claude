"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import {
  deleteRun,
  errorMessage,
  exportCsvUrl,
  getRun,
  getRunDetections,
  getRunEvents,
  getRunTimeseries,
  type Run,
} from "@/lib/api";
import { useAsync } from "@/lib/hooks";
import { fmtDate, fmtMs, fmtSeconds, sortedEntries, sumValues } from "@/lib/format";
import { colorForLabel } from "@/lib/draw";
import { BarList, LineChart } from "./charts";
import { ConfirmDialog } from "./ConfirmDialog";
import { VideoOverlayPlayer } from "./VideoOverlayPlayer";
import { Card, EmptyState, ErrorState, KeyValue, KindBadge, LoadingState, PageHeader, ProgressBar, StatCard, StatusBadge } from "./ui";

/**
 * /runs/[id]
 *   GET /api/runs/{id}              metadata + summary (polled while queued/running)
 *   video & stream runs:            /timeseries, /events  (+ /video and /frames for kind=video)
 *   image runs:                     /detections
 *   GET /api/runs/{id}/export.csv   download;  DELETE /api/runs/{id}
 */
export function RunDetail({ id }: { id: number }) {
  const router = useRouter();
  const run = useAsync((signal) => getRun(id, { signal }), [id]);
  const [confirm, setConfirm] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const r = run.data;
  const inProgress = r?.status === "queued" || r?.status === "running";

  // keep polling while the background job is still working
  const { reload } = run;
  useEffect(() => {
    if (!inProgress) return;
    const t = setTimeout(reload, 1000);
    return () => clearTimeout(t);
  }, [inProgress, r, reload]);

  const onDelete = async () => {
    setDeleting(true);
    setDeleteError(null);
    try {
      await deleteRun(id);
      router.push("/history");
    } catch (e) {
      setDeleteError(errorMessage(e));
      setDeleting(false);
      setConfirm(false);
    }
  };

  if (run.loading && !r) return <LoadingState label={`Loading run #${id}…`} />;
  if (run.error && !r) {
    return (
      <>
        <PageHeader title={`Run #${id}`} />
        {run.errorStatus === 404 ? (
          <EmptyState>
            Run #{id} does not exist (it may have been deleted).{" "}
            <Link href="/history" className="text-accent underline">
              Back to history
            </Link>
          </EmptyState>
        ) : (
          <ErrorState message={run.error} onRetry={run.reload} />
        )}
      </>
    );
  }
  if (!r) return null;

  return (
    <>
      <PageHeader
        title={`Run #${r.id}`}
        description={
          <span className="inline-flex flex-wrap items-center gap-2">
            <KindBadge kind={r.kind} />
            <StatusBadge status={r.status} />
            <span className="truncate">{r.source_name}</span>
          </span>
        }
        actions={
          <>
            <Link href="/history" className="btn">
              Back
            </Link>
            <a className={`btn ${inProgress ? "pointer-events-none opacity-50" : ""}`} href={exportCsvUrl(r.id)} download aria-disabled={inProgress}>
              Export CSV
            </a>
            <button
              type="button"
              className="btn btn-danger"
              onClick={() => setConfirm(true)}
              disabled={inProgress}
              title={inProgress ? "A run cannot be deleted while it is processing" : undefined}
            >
              Delete
            </button>
          </>
        }
      />

      {deleteError && (
        <div className="mb-4">
          <ErrorState message={`Delete failed: ${deleteError}`} />
        </div>
      )}

      {inProgress && (
        <Card className="mb-4">
          <ProgressBar value={r.progress} label={r.status === "queued" ? "Waiting in queue" : "Processing"} />
        </Card>
      )}
      {r.status === "failed" && (
        <div className="mb-4">
          <ErrorState message={`Processing failed: ${r.error ?? "unknown error"}`} />
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-[1fr_20rem]">
        <div className="min-w-0 space-y-4">
          {r.kind === "image" ? <ImageRunBody run={r} /> : <TrackingRunBody run={r} />}
        </div>
        <div className="space-y-4">
          <Card title="Details">
            <KeyValue
              items={[
                ["Model", r.model],
                ["Source", r.source_name || "—"],
                ["Created", fmtDate(r.created_at)],
                ["Resolution", r.width && r.height ? `${r.width}×${r.height}` : "—"],
                ...(r.kind !== "image"
                  ? ([
                      ["FPS", r.fps ? r.fps.toFixed(1) : "—"],
                      ["Frames", r.frames],
                    ] as [string, React.ReactNode][])
                  : []),
                ["Avg latency", fmtMs(r.avg_latency_ms)],
                ["Confidence", r.conf.toFixed(2)],
                ["Counting line", r.line ? r.line.map((v) => v.toFixed(2)).join(", ") : "none"],
              ]}
            />
          </Card>
        </div>
      </div>

      <ConfirmDialog
        open={confirm}
        title={`Delete run #${r.id}?`}
        message="This permanently removes the run, its detections/tracks/events and any stored video. This cannot be undone."
        confirmLabel="Delete run"
        busy={deleting}
        onConfirm={onDelete}
        onCancel={() => setConfirm(false)}
      />
    </>
  );
}

function ImageRunBody({ run }: { run: Run }) {
  const dets = useAsync((signal) => getRunDetections(run.id, { signal }), [run.id]);
  const counts = run.summary.counts ?? {};
  return (
    <>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <StatCard label="Objects" value={sumValues(counts)} />
        <StatCard label="Classes" value={Object.keys(counts).length} />
        <StatCard label="Latency" value={fmtMs(run.summary.timing?.total_ms ?? run.avg_latency_ms)} />
      </div>
      <Card title="Counts per class">
        <BarList data={sortedEntries(counts)} emptyText="Nothing was detected." />
      </Card>
      <Card title="Detections">
        <p className="mb-2 text-xs text-muted">Image runs store detections only, not the image itself.</p>
        {dets.loading && !dets.data ? (
          <LoadingState />
        ) : dets.error ? (
          <ErrorState message={dets.error} onRetry={dets.reload} />
        ) : dets.data && dets.data.length > 0 ? (
          <div className="-mx-4 max-h-[28rem] overflow-auto sm:mx-0">
            <table className="table">
              <thead className="sticky top-0 bg-surface">
                <tr>
                  <th scope="col">#</th>
                  <th scope="col">Label</th>
                  <th scope="col" className="text-right">
                    Score
                  </th>
                  <th scope="col">Box (x1, y1, x2, y2)</th>
                </tr>
              </thead>
              <tbody>
                {dets.data.map((d, i) => (
                  <tr key={i}>
                    <td className="text-muted tabular-nums">{i + 1}</td>
                    <td>
                      <LabelDot label={d.label} />
                    </td>
                    <td className="text-right tabular-nums">{d.score.toFixed(3)}</td>
                    <td className="font-mono text-xs whitespace-nowrap text-muted">{d.box.map((v) => Math.round(v)).join(", ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState>No detections stored.</EmptyState>
        )}
      </Card>
    </>
  );
}

const EVENTS_PAGE = 100;

function TrackingRunBody({ run }: { run: Run }) {
  const done = run.status === "done";
  // timeseries/events are only complete once the job has finished
  const ts = useAsync((signal) => (done ? getRunTimeseries(run.id, { signal }) : Promise.resolve(undefined)), [run.id, done]);
  const events = useAsync((signal) => (done ? getRunEvents(run.id, { signal }) : Promise.resolve(undefined)), [run.id, done]);
  const [eventsShown, setEventsShown] = useState(EVENTS_PAGE);

  const s = run.summary;
  const unique = s.unique_objects ?? {};
  const dwell = s.avg_dwell_seconds ?? {};
  const peak = s.peak_occupancy ?? {};
  const lineIn = s.line?.in ?? {};
  const lineOut = s.line?.out ?? {};

  const series = useMemo(() => (ts.data ? ts.data.labels.map((l) => ({ name: l, values: ts.data?.series[l] ?? [] })) : []), [ts.data]);

  // average dwell over all objects = per-class averages weighted by the number of objects per class
  let dwellNum = 0;
  let dwellDen = 0;
  for (const [label, sec] of Object.entries(dwell)) {
    const n = unique[label] ?? 1;
    dwellNum += sec * n;
    dwellDen += n;
  }
  const avgDwell = dwellDen ? dwellNum / dwellDen : null;

  const labels = Array.from(new Set([...Object.keys(unique), ...Object.keys(peak), ...Object.keys(dwell)])).sort(
    (a, b) => (unique[b] ?? 0) - (unique[a] ?? 0),
  );

  return (
    <>
      {run.kind === "video" && done && (
        <Card title="Playback with tracking overlay">
          <VideoOverlayPlayer runId={run.id} line={run.line} />
        </Card>
      )}
      {run.kind === "stream" && (
        <p className="text-sm text-muted">Live-camera sessions store analytics only — frames are not recorded, so there is no playback.</p>
      )}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard label="Unique objects" value={done ? sumValues(unique) : "…"} />
        <StatCard label="Avg dwell" value={avgDwell !== null ? fmtSeconds(avgDwell) : "—"} />
        <StatCard label="Peak occupancy" value={done ? Math.max(0, ...Object.values(peak)) : "…"} hint="max of any one class" />
        <StatCard label="Crossings" value={run.line ? `${sumValues(lineIn)} / ${sumValues(lineOut)}` : "—"} hint={run.line ? "in / out" : "no counting line"} />
      </div>

      <Card title="Objects in view over time (per-second peak)">
        {!done ? (
          <EmptyState>Available when processing finishes.</EmptyState>
        ) : ts.loading && !ts.data ? (
          <LoadingState />
        ) : ts.error ? (
          <ErrorState message={ts.error} onRetry={ts.reload} />
        ) : (
          <LineChart x={ts.data?.t ?? []} series={series} yLabel="Objects in view" />
        )}
      </Card>

      {labels.length > 0 && (
        <Card title="Per class">
          <div className="-mx-4 overflow-x-auto sm:mx-0">
            <table className="table">
              <thead>
                <tr>
                  <th scope="col">Label</th>
                  <th scope="col" className="text-right">
                    Unique
                  </th>
                  <th scope="col" className="text-right">
                    Avg dwell
                  </th>
                  <th scope="col" className="text-right">
                    Peak
                  </th>
                  {run.line && (
                    <>
                      <th scope="col" className="text-right">
                        In
                      </th>
                      <th scope="col" className="text-right">
                        Out
                      </th>
                    </>
                  )}
                </tr>
              </thead>
              <tbody>
                {labels.map((l) => (
                  <tr key={l}>
                    <td>
                      <LabelDot label={l} />
                    </td>
                    <td className="text-right tabular-nums">{unique[l] ?? 0}</td>
                    <td className="text-right tabular-nums">{dwell[l] !== undefined ? fmtSeconds(dwell[l]) : "—"}</td>
                    <td className="text-right tabular-nums">{peak[l] ?? 0}</td>
                    {run.line && (
                      <>
                        <td className="text-right tabular-nums">{lineIn[l] ?? 0}</td>
                        <td className="text-right tabular-nums">{lineOut[l] ?? 0}</td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <Card title={`Line-crossing events${events.data ? ` (${events.data.length})` : ""}`}>
        {!run.line ? (
          <EmptyState>This run had no counting line.</EmptyState>
        ) : !done ? (
          <EmptyState>Available when processing finishes.</EmptyState>
        ) : events.loading && !events.data ? (
          <LoadingState />
        ) : events.error ? (
          <ErrorState message={events.error} onRetry={events.reload} />
        ) : events.data && events.data.length > 0 ? (
          <>
            <div className="-mx-4 max-h-[28rem] overflow-auto sm:mx-0">
              <table className="table">
                <thead className="sticky top-0 bg-surface">
                  <tr>
                    <th scope="col">Time</th>
                    <th scope="col">Track</th>
                    <th scope="col">Label</th>
                    <th scope="col">Direction</th>
                  </tr>
                </thead>
                <tbody>
                  {events.data.slice(0, eventsShown).map((e, i) => (
                    <tr key={`${e.track_id}-${e.direction}-${i}`}>
                      <td className="tabular-nums">{fmtSeconds(e.t_seconds)}</td>
                      <td className="tabular-nums text-muted">#{e.track_id}</td>
                      <td>
                        <LabelDot label={e.label} />
                      </td>
                      <td>
                        <span className={e.direction === "in" ? "text-success" : "text-warning"}>
                          {e.direction === "in" ? "→ in" : "← out"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {events.data.length > eventsShown && (
              <button type="button" className="btn mt-3" onClick={() => setEventsShown((n) => n + EVENTS_PAGE)}>
                Show more ({events.data.length - eventsShown} left)
              </button>
            )}
          </>
        ) : (
          <EmptyState>No object crossed the line.</EmptyState>
        )}
      </Card>
    </>
  );
}

function LabelDot({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="h-2 w-2 shrink-0 rounded-sm" style={{ background: colorForLabel(label) }} aria-hidden />
      {label}
    </span>
  );
}
