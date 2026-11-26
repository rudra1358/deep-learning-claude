"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Shuffle } from "lucide-react";
import { Controls, Narration, SimShell, fmt, signedColor, useStepper } from "./SimShell";
import {
  canMultiply,
  decodeStep,
  multiply,
  randomIntMatrix,
  resize,
  shapeOf,
  totalSteps,
  type Matrix,
  type StepMode,
} from "@/lib/sim/matmul";

/** Matrices are edited as strings so the user can type "-" or "1." mid-entry. */
type Draft = string[][];

const toDraft = (M: Matrix): Draft => M.map((r) => r.map((v) => String(v)));
const toNum = (D: Draft): Matrix =>
  D.map((r) =>
    r.map((s) => {
      const v = parseFloat(s);
      return Number.isFinite(v) ? v : 0;
    }),
  );

const DEFAULT_A: Matrix = [
  [2, -1, 0],
  [1, 3, -2],
];
const DEFAULT_B: Matrix = [
  [1, 2],
  [0, -1],
  [3, 1],
];

/** Compact display of a number: integers as-is, otherwise up to 2 decimals. */
const show = (v: number) => (Number.isInteger(v) ? String(v) : fmt(v, 2));

type CellState = "none" | "row" | "term" | "done-term";

function MatrixGrid({
  name,
  draft,
  values,
  cellState,
  onEdit,
  outputCell,
}: {
  name: string;
  draft?: Draft;
  values: Matrix;
  cellState: (r: number, c: number) => CellState;
  onEdit?: (r: number, c: number, s: string) => void;
  outputCell?: (r: number, c: number) => { text: string; filled: boolean; current: boolean };
}) {
  const { rows, cols } = shapeOf(values);
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="text-xs text-muted">
        <span className="font-serif text-sm font-semibold text-text">{name}</span> ({rows}×{cols})
      </div>
      <div
        className="grid gap-1 rounded-md border-x-2 border-border-strong px-1 py-0.5"
        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 2.75rem))` }}
        role="group"
        aria-label={`Matrix ${name}, ${rows} by ${cols}`}
      >
        {values.map((row, r) =>
          row.map((v, c) => {
            const st = cellState(r, c);
            const bg =
              st === "term"
                ? "var(--gold-soft)"
                : st === "row" || st === "done-term"
                  ? "color-mix(in srgb, var(--gold-soft) 55%, transparent)"
                  : "transparent";
            const border = st === "term" ? "var(--gold)" : "var(--border)";
            if (outputCell) {
              const o = outputCell(r, c);
              return (
                <div
                  key={`${r}-${c}`}
                  className="flex h-9 items-center justify-center rounded border font-mono text-sm"
                  style={{
                    borderColor: o.current ? "var(--gold)" : "var(--border)",
                    borderWidth: o.current ? 2 : 1,
                    background: o.filled ? signedColor(v, 10) : o.current ? "var(--gold-soft)" : "transparent",
                    color: o.filled || o.current ? "var(--text)" : "var(--text-faint)",
                  }}
                  aria-label={`${name} row ${r + 1} column ${c + 1}: ${o.text}`}
                >
                  {o.text}
                </div>
              );
            }
            return (
              <input
                key={`${r}-${c}`}
                type="text"
                inputMode="decimal"
                value={draft ? draft[r][c] : String(v)}
                onChange={(e) => onEdit?.(r, c, e.target.value)}
                className="h-9 w-full rounded border bg-surface text-center font-mono text-sm text-text"
                style={{ background: bg, borderColor: border, borderWidth: st === "term" ? 2 : 1, fontWeight: st === "term" ? 600 : 400 }}
                aria-label={`${name} row ${r + 1} column ${c + 1}`}
              />
            );
          }),
        )}
      </div>
    </div>
  );
}

function DimSelect({ label, value, max, onChange }: { label: string; value: number; max: number; onChange: (v: number) => void }) {
  return (
    <label className="flex items-center gap-1.5 text-xs text-muted">
      {label}
      <select className="field py-0.5" value={value} onChange={(e) => onChange(+e.target.value)} aria-label={label}>
        {Array.from({ length: max }, (_, i) => i + 1).map((v) => (
          <option key={v} value={v}>
            {v}
          </option>
        ))}
      </select>
    </label>
  );
}

export default function MatmulSim() {
  const [dA, setDA] = useState<Draft>(() => toDraft(DEFAULT_A));
  const [dB, setDB] = useState<Draft>(() => toDraft(DEFAULT_B));
  const [mode, setMode] = useState<StepMode>("term");
  const [seed, setSeed] = useState(7);

  const A = useMemo(() => toNum(dA), [dA]);
  const B = useMemo(() => toNum(dB), [dB]);
  const sA = shapeOf(A);
  const sB = shapeOf(B);
  const ok = canMultiply(A, B);
  const C = useMemo(() => (ok ? multiply(A, B) : []), [A, B, ok]);
  const total = totalSteps(A, B, mode);

  const s = useStepper(total, { baseMs: 1000 });
  const { reset } = s;
  // Any change to shapes, values or step granularity restarts the walk-through.
  useEffect(() => {
    reset();
  }, [dA, dB, mode, reset]);

  const step = Math.min(s.step, total - 1);
  const st = decodeStep(A, B, mode, step);
  const k = sA.cols;

  const edit = (which: "A" | "B") => (r: number, c: number, v: string) => {
    const set = which === "A" ? setDA : setDB;
    set((D) => D.map((row, i) => (i === r ? row.map((x, j) => (j === c ? v : x)) : row)));
  };

  const reshape = (which: "A" | "B", rows: number, cols: number) => {
    const M = which === "A" ? A : B;
    const fresh = randomIntMatrix(rows, cols, seed * 31 + (which === "A" ? 1 : 2));
    const next = toDraft(resize(M, rows, cols, (r, c) => fresh[r][c]));
    (which === "A" ? setDA : setDB)(next);
  };

  const randomize = () => {
    const nextSeed = seed + 1;
    setSeed(nextSeed);
    setDA(toDraft(randomIntMatrix(sA.rows, sA.cols, nextSeed * 2)));
    setDB(toDraft(randomIntMatrix(sB.rows, sB.cols, nextSeed * 2 + 1)));
  };

  // Highlight logic -------------------------------------------------------
  const work = st.kind === "work" ? st : null;
  const aState = (r: number, c: number): CellState => {
    if (!work || r !== work.i) return "none";
    if (c === work.p) return "term";
    return c < work.p ? "done-term" : "row";
  };
  const bState = (r: number, c: number): CellState => {
    if (!work || c !== work.j) return "none";
    if (r === work.p) return "term";
    return r < work.p ? "done-term" : "row";
  };
  const outCell = (r: number, c: number) => {
    const idx = r * sB.cols + c;
    if (st.kind === "done") return { text: show(C[r][c]), filled: true, current: false };
    if (!work) return { text: "·", filled: false, current: false };
    if (idx < work.cell) return { text: show(C[r][c]), filled: true, current: false };
    if (idx === work.cell) return { text: show(work.partial[work.p]), filled: false, current: true };
    return { text: "·", filled: false, current: false };
  };

  // Narration --------------------------------------------------------------
  let title: string;
  let body: ReactNode;
  if (!ok) {
    title = "These shapes don't fit";
    body = (
      <>
        A is {sA.rows}×{sA.cols} and B is {sB.rows}×{sB.cols}. Each output entry pairs up a <b>row of A</b> (length {sA.cols})
        with a <b>column of B</b> (length {sB.rows}). These lengths differ, so they can&apos;t be paired. The <b>inner</b> sizes must
        match: (n×<b>k</b>)·(<b>k</b>×m). Change A&apos;s columns or B&apos;s rows so they are equal.
      </>
    );
  } else if (st.kind === "intro") {
    title = "Ready";
    body = (
      <>
        ({sA.rows}×<b>{sA.cols}</b>)·(<b>{sB.rows}</b>×{sB.cols}) → {sA.rows}×{sB.cols}. The inner sizes match, so C = AB exists. It has{" "}
        {sA.rows * sB.cols} entries. Each is a dot product (multiply pairs, then add) of length {k}: {sA.rows * sB.cols * k} multiplications in total. Press <b>Step</b> or{" "}
        <b>Play</b>.
      </>
    );
  } else if (st.kind === "done") {
    title = "All entries done";
    body = (
      <>
        C is {sA.rows}×{sB.cols}, built from {sA.rows * sB.cols} dot products ({sA.rows * sB.cols * k} multiply-adds). Another way to see it: column j of C
        is A applied to column j of B. So A acts as one transformation applied to every column of B. Edit any entry to watch C update.
      </>
    );
  } else {
    const a = A[st.i][st.p];
    const b = B[st.p][st.j];
    title = `C[${st.i + 1},${st.j + 1}] = row ${st.i + 1} of A · column ${st.j + 1} of B`;
    body =
      mode === "term" ? (
        <>
          Term {st.p + 1} of {k}: A[{st.i + 1},{st.p + 1}]·B[{st.p + 1},{st.j + 1}] = {show(a)}×{show(b)} = <b>{show(st.products[st.p])}</b>. Running
          total so far = <b>{show(st.partial[st.p])}</b>
          {st.p === k - 1 ? ` — done, so C[${st.i + 1},${st.j + 1}] = ${show(st.partial[st.p])}.` : "."}
        </>
      ) : (
        <>
          Multiply the highlighted row and column pair by pair, then add the {k} products: C[{st.i + 1},{st.j + 1}] ={" "}
          <b>{show(st.partial[k - 1])}</b>.
        </>
      );
  }

  const calc = work ? (
    <div className="mt-3 rounded-lg border border-border p-3 text-sm">
      <div className="mb-1.5 text-xs uppercase tracking-wider text-faint">Dot product for C[{work.i + 1},{work.j + 1}]</div>
      <ul className="space-y-0.5 font-mono text-[0.8rem]">
        {work.products.map((v, p) => (
          <li
            key={p}
            style={{
              color: p === work.p ? "var(--gold)" : p < work.p ? "var(--text)" : "var(--text-faint)",
              fontWeight: p === work.p ? 600 : 400,
            }}
          >
            {show(A[work.i][p])} × {show(B[p][work.j])} = {p <= work.p ? show(v) : "?"}
          </li>
        ))}
      </ul>
      <div className="mt-2 border-t border-border pt-1.5 font-mono text-[0.8rem]">
        running sum = <span className="font-semibold text-accent">{show(work.partial[work.p])}</span>
      </div>
    </div>
  ) : null;

  return (
    <SimShell
      title="Matrix Multiplication"
      subtitle="Each entry of C = AB comes from one row of A and one column of B: multiply pairs, then add."
      controls={
        <Controls
          s={s}
          extra={
            <div className="flex flex-wrap items-center gap-2">
              <label className="flex items-center gap-1.5 text-xs text-muted">
                Step by
                <select className="field py-0.5" value={mode} onChange={(e) => setMode(e.target.value as StepMode)} aria-label="Step granularity">
                  <option value="term">product term</option>
                  <option value="cell">output cell</option>
                </select>
              </label>
              <button className="sim-btn" onClick={randomize} aria-label="Randomize matrix entries">
                <Shuffle size={14} /> Randomize
              </button>
            </div>
          }
        />
      }
      aside={
        <>
          <Narration title={title}>{body}</Narration>
          {calc}
        </>
      }
    >
      <div className="mb-4 flex flex-wrap gap-x-5 gap-y-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold">A</span>
          <DimSelect label="rows" value={sA.rows} max={3} onChange={(v) => reshape("A", v, sA.cols)} />
          <DimSelect label="cols" value={sA.cols} max={4} onChange={(v) => reshape("A", sA.rows, v)} />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold">B</span>
          <DimSelect label="rows" value={sB.rows} max={4} onChange={(v) => reshape("B", v, sB.cols)} />
          <DimSelect label="cols" value={sB.cols} max={3} onChange={(v) => reshape("B", sB.rows, v)} />
        </div>
      </div>

      <div className="mb-3 font-mono text-sm" aria-label="Shape check">
        ({sA.rows}×<span style={{ color: ok ? "var(--accent)" : "var(--clay)", fontWeight: 600 }}>{sA.cols}</span>) · (
        <span style={{ color: ok ? "var(--accent)" : "var(--clay)", fontWeight: 600 }}>{sB.rows}</span>×{sB.cols}){" "}
        {ok ? (
          <span>
            → ({sA.rows}×{sB.cols})
          </span>
        ) : (
          <span className="text-clay">→ no match: {sA.cols} ≠ {sB.rows}</span>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <MatrixGrid name="A" draft={dA} values={A} cellState={aState} onEdit={edit("A")} />
        <span className="text-lg text-muted" aria-hidden>
          ×
        </span>
        <MatrixGrid name="B" draft={dB} values={B} cellState={bState} onEdit={edit("B")} />
        {ok && (
          <>
            <span className="text-lg text-muted" aria-hidden>
              =
            </span>
            <MatrixGrid name="C" values={C} cellState={() => "none"} outputCell={outCell} />
          </>
        )}
      </div>
      <p className="mt-4 text-xs text-muted">
        You can edit any entry (decimals and negatives are fine). Gold marks the current row of A, column of B and the product being added. Filled
        cells of C are coloured by sign (accent = positive, clay = negative).
      </p>
    </SimShell>
  );
}
