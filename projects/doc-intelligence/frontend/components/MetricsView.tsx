// Generic renderer for the free-form JSON in /api/metrics.
// The training scripts may add, rename or skip keys, so instead of hard-coding every table we
// follow a few simple rules that turn ANY JSON into readable tables:
//   * plain values (numbers, text)            -> a "key: value" list
//   * an object whose children are objects of numbers (e.g. {bm25: {recall@5: .8}, dense: {...}})
//                                              -> one table: a row per child, a column per metric
//   * a list of objects (e.g. training history) -> a table, a row per item
//   * a list of lists of numbers               -> a matrix (confusion matrix)
//   * anything else                            -> a sub-heading and the same rules again (recursion)
import type { Json } from "@/lib/api";

type Obj = { [key: string]: Json };

const isObj = (v: Json): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
const isScalar = (v: Json) => v === null || typeof v !== "object";

// "Object of numbers": only plain values, none of them text, at least one a number.
// e.g. {"recall@5": 0.88, "mrr@10": 0.75} - several of these side by side make a comparison table.
function isNumberRow(v: Json): v is Obj {
  if (!isObj(v)) return false;
  const values = Object.values(v);
  return values.every((x) => isScalar(x) && typeof x !== "string") && values.some((x) => typeof x === "number");
}

// Columns that are "higher is better" scores get a little bar and the best value in bold.
const SCORE_COLUMN = /recall|mrr|f1|^em$|rouge|accuracy|precision|refused_correctly|contains_gold/i;

function fmt(v: Json): string {
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(Math.abs(v) < 1 ? 4 : 2);
  if (v === null) return "—";
  if (Array.isArray(v) && v.every(isScalar)) return v.map(fmt).join(", ");
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

// Scores are stored either as fractions (0.81) or percentages (81.0); draw both on a 0-100 bar.
function Bar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(100, value <= 1 ? value * 100 : value));
  return (
    <span className="mt-1 block h-1.5 w-full rounded bg-border" aria-hidden="true">
      <span className="block h-full rounded bg-accent" style={{ width: `${pct}%` }} />
    </span>
  );
}

// A table with one row per entry of `rows` and one column per plain value found in any row.
function RowsTable({ rows, rowHeader }: { rows: [string, Obj][]; rowHeader: string }) {
  const columns: string[] = [];
  // plain values and short lists (e.g. several gold answers) become columns; deeper structures are skipped
  for (const [, r] of rows)
    for (const [k, v] of Object.entries(r))
      if ((isScalar(v) || (Array.isArray(v) && v.every(isScalar))) && !columns.includes(k)) columns.push(k);

  // best value of each score column, to make the winner bold
  const best: Record<string, number> = {};
  for (const c of columns.filter((c) => SCORE_COLUMN.test(c)))
    best[c] = Math.max(...rows.map(([, r]) => (typeof r[c] === "number" ? (r[c] as number) : -Infinity)));

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="text-xs text-muted">
          <tr className="border-b border-border">
            <th className="p-2">{rowHeader}</th>
            {columns.map((c) => (
              <th key={c} className="p-2 text-right">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, r]) => (
            <tr key={name} className="border-b border-border last:border-0">
              <th className="p-2 font-medium">{name}</th>
              {columns.map((c) => {
                const v = r[c] ?? null;
                const score = SCORE_COLUMN.test(c) && typeof v === "number";
                return (
                  <td key={c} className={`min-w-16 p-2 text-right align-top ${score && rows.length > 1 && v === best[c] ? "font-bold" : ""}`}>
                    <span className="break-words">{fmt(v)}</span>
                    {score && <Bar value={v} />}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Matrix({ data, labels }: { data: number[][]; labels?: string[] }) {
  const max = Math.max(1, ...data.flat());
  return (
    <div className="overflow-x-auto">
      <table className="text-right text-xs">
        <caption className="mb-1 text-left text-muted">rows = true class, columns = predicted class</caption>
        <thead>
          <tr>
            <th />
            {data[0]?.map((_, j) => (
              <th key={j} className="p-1.5 font-medium text-muted">
                {labels?.[j] ?? j}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i}>
              <th className="p-1.5 text-left font-medium text-muted">{labels?.[i] ?? i}</th>
              {row.map((v, j) => (
                // cell shade grows with the count, so the diagonal (correct predictions) stands out
                <td key={j} className="p-1.5" style={{ background: `color-mix(in srgb, var(--accent) ${Math.round((v / max) * 60)}%, transparent)` }}>
                  {v.toLocaleString()}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const MAX_LIST_ROWS = 5; // long lists (e.g. example answers) are cut so the page stays readable

export function JsonBlock({ value, labels, depth = 0 }: { value: Json; labels?: string[]; depth?: number }) {
  if (isScalar(value)) return <span>{fmt(value)}</span>;

  if (Array.isArray(value)) {
    if (value.every(isScalar)) return <span>{fmt(value)}</span>;
    if (value.every((r) => Array.isArray(r) && r.every((x) => typeof x === "number")))
      return <Matrix data={value as number[][]} labels={labels} />;
    if (value.every(isObj)) {
      const rows = value.slice(0, MAX_LIST_ROWS).map((r, i): [string, Obj] => [String(i + 1), r as Obj]);
      return (
        <>
          <RowsTable rows={rows} rowHeader="#" />
          {value.length > MAX_LIST_ROWS && <p className="text-xs text-muted">… and {value.length - MAX_LIST_ROWS} more</p>}
        </>
      );
    }
    return <pre className="overflow-x-auto text-xs">{JSON.stringify(value, null, 1)}</pre>;
  }

  // An object: split its entries by the rules described at the top of the file.
  const entries = Object.entries(value);
  const plain = entries.filter(([k, v]) => (isScalar(v) || (Array.isArray(v) && v.every(isScalar))) && k !== "labels");
  const rows = entries.filter(([, v]) => isNumberRow(v)) as [string, Obj][];
  const nested = entries.filter(([k, v]) => !plain.some(([p]) => p === k) && !isNumberRow(v) && k !== "labels");
  const labelList = Array.isArray(value.labels) && value.labels.every((x) => typeof x === "string") ? (value.labels as string[]) : labels;

  return (
    <div className={`space-y-3 ${depth > 0 ? "border-l border-border pl-3" : ""}`}>
      {plain.length > 0 && (
        <dl className="grid grid-cols-[minmax(0,auto)_1fr] gap-x-4 gap-y-1 text-sm">
          {plain.map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-muted">{k}</dt>
              <dd className="min-w-0 break-words">{fmt(v)}</dd>
            </div>
          ))}
        </dl>
      )}
      {rows.length > 0 && <RowsTable rows={rows} rowHeader="" />}
      {nested.map(([k, v]) => (
        <div key={k}>
          <h4 className="mb-1 text-sm font-semibold">{k}</h4>
          <JsonBlock value={v} labels={labelList} depth={depth + 1} />
        </div>
      ))}
    </div>
  );
}
