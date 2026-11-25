import { Link, useNavigate } from "react-router-dom";
import { fmtDate, fmtMs, shortId, sortedEntries, sumValues } from "../lib/format";
import { KindBadge, StatusBadge } from "./ui";

/** Objects found in a run: detections for images, unique tracked objects for video/stream. */
export function runObjectCount(r) {
  return r.kind === "image" ? sumValues(r.summary.counts) : sumValues(r.summary.unique_objects);
}

function topLabels(r) {
  const rec = r.kind === "image" ? r.summary.counts : r.summary.unique_objects;
  return sortedEntries(rec)
    .slice(0, 3)
    .map(([k, v]) => `${k} ${v}`)
    .join(", ");
}

/** Runs table; the whole row is clickable, the file name is the accessible link. */
export function RunsTable({ runs, compact }) {
  const navigate = useNavigate();
  return (
    <div className="-mx-4 overflow-x-auto sm:mx-0">
      <table className="table min-w-[36rem]">
        <thead>
          <tr>
            <th scope="col">Run</th>
            <th scope="col">Kind</th>
            <th scope="col">Status</th>
            <th scope="col">Model</th>
            <th scope="col" className="text-right">
              Objects
            </th>
            {!compact && <th scope="col">Top labels</th>}
            {!compact && (
              <th scope="col" className="text-right">
                Latency
              </th>
            )}
            <th scope="col">Created</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id} className="cursor-pointer hover:bg-surface-2" onClick={() => navigate(`/runs/${r.id}`)}>
              <td className="max-w-[14rem]">
                <Link
                  to={`/runs/${r.id}`}
                  className="block truncate font-medium hover:underline"
                  onClick={(e) => e.stopPropagation()}
                >
                  #{shortId(r.id)} {r.source_name || "—"}
                </Link>
              </td>
              <td>
                <KindBadge kind={r.kind} />
              </td>
              <td>
                <StatusBadge status={r.status} />
              </td>
              <td className="whitespace-nowrap text-muted">{r.model}</td>
              <td className="text-right tabular-nums">{runObjectCount(r)}</td>
              {!compact && <td className="max-w-[16rem] truncate text-muted">{topLabels(r) || "—"}</td>}
              {!compact && <td className="text-right whitespace-nowrap tabular-nums">{fmtMs(r.avg_latency_ms)}</td>}
              <td className="whitespace-nowrap text-muted">{fmtDate(r.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
