"use client";

import { useEffect, useMemo, useState, type CSSProperties, type ReactNode } from "react";
import { Shuffle } from "lucide-react";
import { Controls, Narration, SimShell, useStepper } from "./SimShell";
import {
  A_RANGE,
  B_RANGE,
  OPS,
  PRESETS,
  align,
  broadcastApply,
  comparePairs,
  getAt,
  idxStr,
  isRepeat,
  numpyError,
  opSymbol,
  ravel,
  seededArray,
  shapeStr,
  sizeOf,
  sourceIndex,
  type DimPair,
  type NDArray,
  type Op,
  type Shape,
} from "@/lib/sim/broadcasting";

/* ------------------------------------------------------------------ */
/* Phases of the walk-through                                          */
/* ------------------------------------------------------------------ */

type Phase =
  | { kind: "align" }
  | { kind: "compare"; w: number } // w = how many pairs (counted from the right) are being / have been checked, 0-based current
  | { kind: "shape" }
  | { kind: "stretch" }
  | { kind: "result" };

/* ------------------------------------------------------------------ */
/* Small presentational pieces                                         */
/* ------------------------------------------------------------------ */

type CellLook = { value: number; repeat?: boolean; target?: boolean; source?: boolean };

function Cell({ c, onHover, label }: { c: CellLook; onHover?: () => void; label: string }) {
  const style: CSSProperties = {
    borderColor: c.target ? "var(--gold)" : "var(--border-strong)",
    borderWidth: c.target ? 2 : 1,
    borderStyle: c.repeat ? "dashed" : "solid",
    background: c.source ? "var(--gold-soft)" : c.repeat ? "transparent" : "var(--surface)",
    color: c.repeat && !c.target ? "var(--text-faint)" : "var(--text)",
    opacity: c.repeat && !c.target ? 0.55 : 1,
    fontWeight: c.target || c.source ? 600 : 400,
  };
  const cls = "flex h-7 w-7 items-center justify-center rounded font-mono text-xs";
  if (onHover)
    return (
      <button type="button" className={cls} style={style} onMouseEnter={onHover} onFocus={onHover} onClick={onHover} aria-label={label}>
        {c.value}
      </button>
    );
  return (
    <div className={cls} style={style} aria-label={label}>
      {c.value}
    </div>
  );
}

/**
 * Draw an array of 1–3 axes: 1-D as one row, 2-D as a grid, 3-D as a row of 2-D slices.
 * `look(idx)` decides how each cell is drawn.
 */
function ArrayView({
  name,
  shape,
  caption,
  look,
  onHover,
}: {
  name: string;
  shape: Shape;
  caption?: ReactNode;
  look: (idx: number[]) => CellLook;
  onHover?: (idx: number[]) => void;
}) {
  const nd = shape.length;
  const slices = nd === 3 ? shape[0] : 1;
  const rows = nd === 1 ? 1 : shape[nd - 2];
  const cols = shape[nd - 1];
  const full = (k: number, r: number, c: number) => (nd === 1 ? [c] : nd === 2 ? [r, c] : [k, r, c]);
  return (
    <div className="min-w-0">
      <div className="mb-1 text-xs text-muted">
        <span className="font-serif text-sm font-semibold text-text">{name}</span> {shapeStr(shape)}
        {caption && <span className="ml-1">{caption}</span>}
      </div>
      <div className="flex flex-wrap gap-3">
        {Array.from({ length: slices }, (_, k) => (
          <div key={k} className="min-w-0">
            {nd === 3 && <div className="mb-0.5 font-mono text-[0.65rem] text-faint">[{k}, :, :]</div>}
            <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${cols}, 1.75rem)` }} role="group" aria-label={`${name} ${nd === 3 ? `slice ${k}` : ""}`}>
              {Array.from({ length: rows * cols }, (_, f) => {
                const r = Math.floor(f / cols);
                const c = f % cols;
                const idx = full(k, r, c);
                return <Cell key={f} c={look(idx)} label={`${name}${idxStr(idx)} = ${look(idx).value}`} onHover={onHover ? () => onHover(idx) : undefined} />;
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ShapeEditor({ name, shape, onChange }: { name: string; shape: Shape; onChange: (s: Shape) => void }) {
  const setNd = (nd: number) => {
    if (nd === shape.length) return;
    onChange(nd > shape.length ? [...new Array(nd - shape.length).fill(2), ...shape] : shape.slice(shape.length - nd));
  };
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs text-muted">
      <span className="font-semibold text-text">{name}</span>
      <label className="flex items-center gap-1">
        axes
        <select className="field py-0.5" value={shape.length} onChange={(e) => setNd(+e.target.value)} aria-label={`${name} number of axes`}>
          {[1, 2, 3].map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      </label>
      <span className="font-mono">(</span>
      {shape.map((d, k) => (
        <select
          key={k}
          className="field py-0.5"
          value={d}
          onChange={(e) => onChange(shape.map((x, j) => (j === k ? +e.target.value : x)))}
          aria-label={`${name} size of axis ${k}`}
        >
          {[1, 2, 3, 4, 5].map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      ))}
      <span className="font-mono">{shape.length === 1 ? ",)" : ")"}</span>
    </div>
  );
}

const ruleShort = (p: DimPair) =>
  p.rule === "equal" ? "equal ✓" : p.rule === "a-is-1" ? "A is 1 ✓" : p.rule === "b-is-1" ? "B is 1 ✓" : "clash ✗";

function ruleSentence(p: DimPair): ReactNode {
  const a = p.a ?? 1;
  const b = p.b ?? 1;
  const miss = (who: string, v: number | null) => (v === null ? <> ({who} has no axis here, so it counts as 1)</> : null);
  switch (p.rule) {
    case "equal":
      return (
        <>
          Sizes {a} and {b} are <b>equal</b>
          {miss("A", p.a)}
          {miss("B", p.b)}, so the result keeps size <b>{p.out}</b> on this axis.
        </>
      );
    case "a-is-1":
      return (
        <>
          A has size <b>1</b>
          {miss("A", p.a)} and B has {b}, so A gets <b>stretched</b> along this axis: whatever A holds here is reused {b} times. Result size <b>{p.out}</b>.
        </>
      );
    case "b-is-1":
      return (
        <>
          B has size <b>1</b>
          {miss("B", p.b)} and A has {a}, so B gets <b>stretched</b> along this axis: whatever B holds here is reused {a} times. Result size <b>{p.out}</b>.
        </>
      );
    default:
      return (
        <>
          Sizes <b>{a}</b> and <b>{b}</b> are different, and <b>neither is 1</b>. NumPy has no way to pair {a} elements with {b} elements, so
          broadcasting fails here.
        </>
      );
  }
}

/* ------------------------------------------------------------------ */
/* The simulator                                                       */
/* ------------------------------------------------------------------ */

export default function BroadcastingSim() {
  const [aShape, setAShape] = useState<Shape>(PRESETS[0].a);
  const [bShape, setBShape] = useState<Shape>(PRESETS[0].b);
  const [op, setOp] = useState<Op>(PRESETS[0].op);
  const [preset, setPreset] = useState<string>(PRESETS[0].id);
  const [seed, setSeed] = useState(1);
  const [sel, setSel] = useState<number[]>([]);

  const A: NDArray = useMemo(() => seededArray(aShape, seed * 101 + 7, A_RANGE[0], A_RANGE[1]), [aShape, seed]);
  const B: NDArray = useMemo(() => seededArray(bShape, seed * 211 + 13, B_RANGE[0], B_RANGE[1]), [bShape, seed]);
  const C = useMemo(() => broadcastApply(op, A, B), [op, A, B]);

  const al = align(aShape, bShape);
  const pairs = comparePairs(aShape, bShape);
  const walk = [...pairs].reverse(); // trailing axis first
  const badW = walk.findIndex((p) => p.rule === "incompatible");
  const total = badW >= 0 ? 1 + badW + 1 : 1 + al.rank + 3;

  const s = useStepper(total, { baseMs: 1300 });
  const { reset } = s;
  useEffect(() => {
    reset();
    setSel([]);
  }, [aShape, bShape, op, seed, reset]);

  const step = Math.min(s.step, total - 1);
  const phase: Phase =
    step === 0
      ? { kind: "align" }
      : step <= al.rank
        ? { kind: "compare", w: step - 1 }
        : step === al.rank + 1
          ? { kind: "shape" }
          : step === al.rank + 2
            ? { kind: "stretch" }
            : { kind: "result" };

  const checkedCols = new Set<number>();
  const curCol = phase.kind === "compare" ? walk[phase.w].col : -1;
  if (phase.kind === "compare") for (let w = 0; w <= phase.w; w++) checkedCols.add(walk[w].col);
  else if (phase.kind !== "align") pairs.forEach((p) => checkedCols.add(p.col));
  const failed = phase.kind === "compare" && walk[phase.w].rule === "incompatible";

  const outShape = C?.shape ?? null;
  const showStretched = (phase.kind === "stretch" || phase.kind === "result") && outShape !== null;
  const selIdx = phase.kind === "result" && outShape ? (sel.length === outShape.length ? sel : outShape.map(() => 0)) : null;

  const applyPreset = (id: string) => {
    setPreset(id);
    const p = PRESETS.find((x) => x.id === id);
    if (!p) return;
    setAShape(p.a);
    setBShape(p.b);
    setOp(p.op);
  };
  const editShape = (which: "A" | "B") => (sh: Shape) => {
    setPreset("custom");
    (which === "A" ? setAShape : setBShape)(sh);
  };

  /* ---------------- narration ---------------- */
  let title = "";
  let body: ReactNode = null;
  const presetNote = PRESETS.find((p) => p.id === preset)?.note;
  if (phase.kind === "align") {
    title = "Step 1: line up the shapes on the right";
    body = (
      <>
        A has shape <b>{shapeStr(aShape)}</b> and B has shape <b>{shapeStr(bShape)}</b>. Write them so their <b>last</b> axes line up.
        {aShape.length !== bShape.length && (
          <>
            {" "}
            The shorter shape is missing {Math.abs(aShape.length - bShape.length)} axis(es) on the left. Broadcasting treats a missing axis as size 1
            (shown as ·).
          </>
        )}{" "}
        Then we compare sizes column by column, starting from the right.
        {presetNote && <span className="mt-1 block">{presetNote}</span>}
      </>
    );
  } else if (phase.kind === "compare") {
    const p = walk[phase.w];
    const axisName = `axis ${p.col - al.rank}`;
    title = failed ? `Step 2: ${axisName} does not fit, stop` : `Step 2: compare ${axisName} (${phase.w + 1} of ${al.rank})`;
    body = failed ? (
      <>
        {ruleSentence(p)}
        <span className="mt-2 block font-mono text-[0.78rem] text-clay">{numpyError(aShape, bShape)}</span>
        <span className="mt-1 block">
          To fix it, make the sizes match, or add a size-1 axis where you mean &quot;repeat&quot;. For example, <code>b[:, None]</code> turns (3,) into
          (3, 1).
        </span>
      </>
    ) : (
      <>
        The rule: two sizes fit if they are <b>equal</b> or <b>one of them is 1</b>. {ruleSentence(p)}
      </>
    );
  } else if (phase.kind === "shape" && outShape) {
    title = "Step 3: the result shape";
    body = (
      <>
        Every column passed. The result takes the larger size in each column: <b>{shapeStr(outShape)}</b>, with {sizeOf(outShape)} elements.
        {sizeOf(outShape) > Math.max(sizeOf(aShape), sizeOf(bShape)) && (
          <> Notice it is bigger than both inputs ({sizeOf(aShape)} and {sizeOf(bShape)} elements).</>
        )}
      </>
    );
  } else if (phase.kind === "stretch" && outShape) {
    title = "Step 4: stretch the size-1 axes";
    body = (
      <>
        Both arrays are now drawn at the result shape {shapeStr(outShape)}. Solid cells are the numbers really stored. <b>Faded dashed cells are repeats</b>{" "}
        of a stored number along a size-1 (or missing) axis. <b>No memory is copied</b>: NumPy steps along that axis by 0 (a stride of 0), so it keeps
        re-reading the same number. A stores {sizeOf(aShape)} numbers and B stores {sizeOf(bShape)}.
      </>
    );
  } else if (phase.kind === "result" && outShape && C && selIdx) {
    const ia = sourceIndex(selIdx, aShape);
    const ib = sourceIndex(selIdx, bShape);
    const va = getAt(A, ia);
    const vb = getAt(B, ib);
    title = "Step 5: the element-wise result";
    body = (
      <>
        Hover over or tap any cell of the result. For C{idxStr(selIdx)}:
        <span className="mt-1 block font-mono text-[0.82rem] text-text">
          C{idxStr(selIdx)} = A{idxStr(ia)} {opSymbol(op)} B{idxStr(ib)} = {va} {opSymbol(op)} {vb} = <b>{getAt(C, selIdx)}</b>
        </span>
        <span className="mt-1 block">
          To find where each number comes from: drop the leading axes the array does not have, and use index <b>0</b> on every axis where it has size 1.
        </span>
      </>
    );
  }

  /* ---------------- cell looks ---------------- */
  const lookOriginal = (arr: NDArray) => (idx: number[]): CellLook => ({ value: getAt(arr, idx) });
  const lookStretched = (arr: NDArray) => (idx: number[]): CellLook => {
    const src = sourceIndex(idx, arr.shape);
    const look: CellLook = { value: getAt(arr, src), repeat: isRepeat(idx, arr.shape) };
    if (selIdx) {
      look.target = ravel(idx, outShape as Shape) === ravel(selIdx, outShape as Shape);
      // the stored element that feeds the selected output: the non-repeat position with the same source index
      look.source = !look.repeat && ravel(src, arr.shape) === ravel(sourceIndex(selIdx, arr.shape), arr.shape);
    }
    return look;
  };

  /* ---------------- alignment table ---------------- */
  const cellBg = (col: number) =>
    col === curCol ? (failed ? "color-mix(in srgb, var(--clay) 18%, transparent)" : "var(--gold-soft)") : "transparent";
  const fmtDim = (v: number | null) => (v === null ? "·" : String(v));

  return (
    <SimShell
      title="Broadcasting"
      subtitle="How NumPy combines arrays of different shapes: line them up from the right, then stretch size-1 axes without copying."
      controls={
        <Controls
          s={s}
          extra={
            <div className="flex flex-wrap items-center gap-2">
              <label className="flex items-center gap-1.5 text-xs text-muted">
                Preset
                <select className="field max-w-[11rem] py-0.5" value={preset} onChange={(e) => applyPreset(e.target.value)} aria-label="Preset example">
                  {PRESETS.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.label}
                    </option>
                  ))}
                  <option value="custom">custom shapes</option>
                </select>
              </label>
              <button className="sim-btn" onClick={() => setSeed((v) => v + 1)} aria-label="Draw new random values">
                <Shuffle size={14} /> New values
              </button>
            </div>
          }
        />
      }
      aside={<Narration title={title}>{body}</Narration>}
    >
      {/* shape + op editors */}
      <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2">
        <ShapeEditor name="A" shape={aShape} onChange={editShape("A")} />
        <label className="flex items-center gap-1.5 text-xs text-muted">
          op
          <select
            className="field py-0.5"
            value={op}
            onChange={(e) => {
              setPreset("custom");
              setOp(e.target.value as Op);
            }}
            aria-label="Element-wise operation"
          >
            {OPS.map((o) => (
              <option key={o.op} value={o.op}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
        <ShapeEditor name="B" shape={bShape} onChange={editShape("B")} />
      </div>

      {/* step 1-3: aligned shape table */}
      <div className="mb-4 max-w-full overflow-x-auto">
        <table className="border-collapse font-mono text-xs" aria-label="Right-aligned shapes">
          <thead>
            <tr className="text-faint">
              <th className="px-2 py-1 text-left font-normal"> </th>
              {al.a.map((_, col) => (
                <th key={col} className="px-2 py-1 text-center font-normal" style={{ background: cellBg(col) }}>
                  axis {col - al.rank}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(["A", "B"] as const).map((who) => (
              <tr key={who}>
                <td className="px-2 py-1 font-semibold">
                  {who} {shapeStr(who === "A" ? aShape : bShape)}
                </td>
                {(who === "A" ? al.a : al.b).map((v, col) => (
                  <td
                    key={col}
                    className="border border-border px-3 py-1 text-center"
                    style={{ background: cellBg(col), color: v === null ? "var(--text-faint)" : "var(--text)" }}
                  >
                    {fmtDim(v)}
                  </td>
                ))}
              </tr>
            ))}
            <tr>
              <td className="px-2 py-1 text-muted">check</td>
              {pairs.map((p) => (
                <td
                  key={p.col}
                  className="border border-border px-2 py-1 text-center text-[0.7rem]"
                  style={{
                    background: cellBg(p.col),
                    color: !checkedCols.has(p.col) ? "var(--text-faint)" : p.rule === "incompatible" ? "var(--clay)" : "var(--accent)",
                  }}
                >
                  {checkedCols.has(p.col) ? ruleShort(p) : "?"}
                </td>
              ))}
            </tr>
            <tr>
              <td className="px-2 py-1 font-semibold">result</td>
              {pairs.map((p) => (
                <td key={p.col} className="border border-border px-3 py-1 text-center font-semibold" style={{ background: cellBg(p.col) }}>
                  {checkedCols.has(p.col) ? (p.out ?? "✗") : "?"}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
        {(phase.kind === "shape" || showStretched) && outShape && (
          <div className="mt-1.5 font-mono text-xs">
            {shapeStr(aShape)} {opSymbol(op)} {shapeStr(bShape)} → <span className="font-semibold text-accent">{shapeStr(outShape)}</span>
          </div>
        )}
        {failed && <div className="mt-1.5 font-mono text-xs text-clay">{numpyError(aShape, bShape)}</div>}
      </div>

      {/* arrays */}
      {!showStretched ? (
        <div className="flex flex-wrap items-start gap-x-6 gap-y-3">
          <ArrayView name="A" shape={aShape} look={lookOriginal(A)} caption="as stored" />
          <ArrayView name="B" shape={bShape} look={lookOriginal(B)} caption="as stored" />
        </div>
      ) : (
        outShape && (
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap items-start gap-x-6 gap-y-3">
              <ArrayView name="A" shape={outShape} look={lookStretched(A)} caption={`stretched from ${shapeStr(aShape)}`} />
              <ArrayView name="B" shape={outShape} look={lookStretched(B)} caption={`stretched from ${shapeStr(bShape)}`} />
            </div>
            {phase.kind === "result" && C && selIdx && (
              <ArrayView
                name={`C = A ${opSymbol(op)} B`}
                shape={outShape}
                look={(idx) => ({ value: getAt(C, idx), target: ravel(idx, outShape) === ravel(selIdx, outShape) })}
                onHover={(idx) => setSel(idx)}
              />
            )}
            <p className="text-xs text-muted">
              Dashed, faded cells are pretend repeats. No memory is copied. {phase.kind === "result" && "Gold outline: the cell that lines up with the chosen output. Gold fill: the stored number it reads."}
            </p>
          </div>
        )
      )}
      {phase.kind === "result" && C && (
        <p className="mt-2 font-mono text-[0.7rem] text-faint">
          memory: A stores {sizeOf(aShape)}, B stores {sizeOf(bShape)}. Only the {sizeOf(C.shape)} results are new.
        </p>
      )}
    </SimShell>
  );
}
