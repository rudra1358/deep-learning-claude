"use client";

import { useEffect, useMemo, useState, type MouseEvent } from "react";
import { Controls, Narration, SimShell, Slider, signedColor, seqColor, useStepper } from "./SimShell";
import { conv2d, convStep, flipKernel, outputSize, pad2d, pool2d, relu, zeros, type Grid } from "@/lib/sim/conv";

/* ------------------------------------------------------------------ */
/* Presets                                                             */
/* ------------------------------------------------------------------ */

const LESSON_INPUT: Grid = [
  [1, 2, 0, 1, 3],
  [0, 1, 3, 2, 1],
  [2, 1, 0, 1, 0],
  [1, 3, 1, 0, 2],
  [0, 2, 1, 3, 1],
];

const DIGIT_3 = [".###.", "#...#", "....#", "..##.", "....#", "#...#", ".###."];

type InputPreset = "lesson" | "vedge" | "hedge" | "diagonal" | "checker" | "digit" | "ramp" | "blank";

const INPUT_PRESETS: { id: InputPreset; label: string }[] = [
  { id: "lesson", label: "Lesson example (5×5)" },
  { id: "vedge", label: "Vertical edge" },
  { id: "hedge", label: "Horizontal edge" },
  { id: "diagonal", label: "Diagonal line" },
  { id: "checker", label: "Checkerboard" },
  { id: "digit", label: "Digit “3”" },
  { id: "ramp", label: "Brightness ramp" },
  { id: "blank", label: "Blank (draw your own)" },
];

function makeInput(preset: InputPreset, n: number): Grid {
  const g = zeros(n, n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      switch (preset) {
        case "lesson":
          g[i][j] = i < 5 && j < 5 ? LESSON_INPUT[i][j] : 0;
          break;
        case "vedge":
          g[i][j] = j < Math.floor(n / 2) ? 0 : 9;
          break;
        case "hedge":
          g[i][j] = i < Math.floor(n / 2) ? 9 : 0;
          break;
        case "diagonal":
          g[i][j] = i === j ? 9 : 0;
          break;
        case "checker":
          g[i][j] = (Math.floor(i / 2) + Math.floor(j / 2)) % 2 === 0 ? 9 : 0;
          break;
        case "digit": {
          const top = Math.floor((n - DIGIT_3.length) / 2);
          const left = Math.floor((n - 5) / 2);
          const row = DIGIT_3[i - top];
          g[i][j] = row && j - left >= 0 && j - left < 5 && row[j - left] === "#" ? 9 : 0;
          break;
        }
        case "ramp":
          g[i][j] = Math.round((9 * j) / (n - 1));
          break;
        case "blank":
          g[i][j] = 0;
          break;
      }
    }
  return g;
}

type KernelPreset = "lesson" | "identity" | "sobelX" | "sobelY" | "box" | "gauss" | "sharpen" | "laplacian" | "custom";

const KERNELS: Record<Exclude<KernelPreset, "custom">, { label: string; k: Grid; note: string }> = {
  lesson: { label: "Vertical edge [1 0 −1] (lesson)", k: [[1, 0, -1], [1, 0, -1], [1, 0, -1]], note: "Left column minus right column: large positive where the image gets darker from left to right, negative where it gets brighter." },
  identity: { label: "Identity", k: [[0, 0, 0], [0, 1, 0], [0, 0, 0]], note: "Copies the centre pixel: the output is the input (cropped by the borders when p = 0)." },
  sobelX: { label: "Sobel — vertical edges", k: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], note: "Horizontal derivative with extra weight on the centre row. Responds to vertical edges, positive for dark → bright (left to right)." },
  sobelY: { label: "Sobel — horizontal edges", k: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], note: "Vertical derivative. Responds to horizontal edges, positive for dark → bright (top to bottom)." },
  box: { label: "Box blur (1/9)", k: [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map((r) => r.map((v) => v / 9)), note: "Average of the 3×3 neighbourhood: smooths noise, softens edges. Weights sum to 1 so flat regions keep their brightness." },
  gauss: { label: "Gaussian blur (1/16)", k: [[1, 2, 1], [2, 4, 2], [1, 2, 1]].map((r) => r.map((v) => v / 16)), note: "Weighted average that trusts the centre most — a smoother blur than the box filter. Weights sum to 1." },
  sharpen: { label: "Sharpen", k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], note: "Identity plus a negative Laplacian: amplifies a pixel's difference from its neighbours. Weights sum to 1." },
  laplacian: { label: "Laplacian", k: [[0, 1, 0], [1, -4, 1], [0, 1, 0]], note: "Discrete second derivative: zero on flat regions and on linear ramps, large on edges, corners and isolated dots. Weights sum to 0." },
};

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

/** Compact number formatting for cells: integers stay integers, otherwise 1–2 decimals. */
function nf(v: number): string {
  const r = Math.round(v * 100) / 100;
  const s = Number.isInteger(r) ? String(r) : Math.abs(r) >= 10 ? r.toFixed(1) : r.toFixed(2);
  return s.replace("-", "−");
}

const maxAbs = (g: Grid) => g.reduce((m, r) => r.reduce((mm, v) => Math.max(mm, Math.abs(v)), m), 0) || 1;

interface Rect {
  r: number;
  c: number;
  h: number;
  w: number;
}

/** A numeric grid drawn as SVG cells. */
function GridSvg({
  values,
  label,
  cs = 30,
  pad = 0,
  color,
  shown,
  highlight,
  onCell,
  onCellAlt,
  maxWidth,
}: {
  values: Grid;
  label: string;
  cs?: number;
  pad?: number;
  color: (v: number) => { fill: string; strong: boolean };
  shown?: (i: number, j: number) => boolean;
  highlight?: Rect | null;
  onCell?: (i: number, j: number, e: MouseEvent) => void;
  onCellAlt?: (i: number, j: number) => void;
  maxWidth: number;
}) {
  const h = values.length;
  const w = values[0]?.length ?? 0;
  const fs = Math.max(9, Math.min(13, cs * 0.4));
  return (
    <svg viewBox={`-2 -2 ${w * cs + 4} ${h * cs + 4}`} className="h-auto w-full" style={{ maxWidth }} role="img" aria-label={label}>
      {values.map((row, i) =>
        row.map((v, j) => {
          const isPad = i < pad || j < pad || i >= h - pad || j >= w - pad;
          const visible = shown ? shown(i, j) : true;
          const col = visible && !isPad ? color(v) : null;
          return (
            <g
              key={`${i}-${j}`}
              onClick={onCell && !isPad ? (e) => onCell(i - pad, j - pad, e) : undefined}
              onContextMenu={
                onCellAlt && !isPad
                  ? (e) => {
                      e.preventDefault();
                      onCellAlt(i - pad, j - pad);
                    }
                  : undefined
              }
              style={{ cursor: onCell && !isPad ? "pointer" : "default" }}
            >
              <rect
                x={j * cs}
                y={i * cs}
                width={cs}
                height={cs}
                fill={isPad ? "var(--surface-2)" : col ? col.fill : "var(--surface)"}
                stroke={isPad ? "var(--text-faint)" : "var(--border)"}
                strokeDasharray={isPad ? "3 2" : undefined}
                strokeWidth={1}
              />
              {(visible || isPad) && (
                <text
                  x={j * cs + cs / 2}
                  y={i * cs + cs / 2 + fs * 0.36}
                  textAnchor="middle"
                  fontSize={fs}
                  fontFamily="var(--font-mono, monospace)"
                  fill={isPad ? "var(--text-faint)" : col?.strong ? "var(--surface)" : "var(--text)"}
                  style={{ pointerEvents: "none" }}
                >
                  {nf(v)}
                </text>
              )}
            </g>
          );
        }),
      )}
      {highlight && (
        <rect
          x={highlight.c * cs}
          y={highlight.r * cs}
          width={highlight.w * cs}
          height={highlight.h * cs}
          fill="none"
          stroke="var(--gold)"
          strokeWidth={3}
          rx={2}
          style={{ transition: "x 0.25s, y 0.25s" }}
        />
      )}
    </svg>
  );
}

/** Text input for one kernel weight; keeps a local draft so "-" or "0." can be typed. */
function NumCell({ value, onCommit, background, label }: { value: number; onCommit: (v: number) => void; background: string; label: string }) {
  const [draft, setDraft] = useState(() => String(Math.round(value * 1000) / 1000));
  useEffect(() => {
    setDraft((d) => (parseFloat(d) === value ? d : String(Math.round(value * 1000) / 1000)));
  }, [value]);
  return (
    <input
      type="text"
      inputMode="decimal"
      value={draft}
      onChange={(e) => {
        setDraft(e.target.value);
        const v = parseFloat(e.target.value);
        if (Number.isFinite(v)) onCommit(v);
      }}
      className="w-full rounded border border-border px-0.5 py-1 text-center font-mono text-xs"
      style={{ background, color: "var(--text)" }}
      aria-label={label}
    />
  );
}

const seqCell =(max: number) => (v: number) => ({ fill: seqColor(v / max), strong: v / max > 0.6 });
const signedCell = (max: number) => (v: number) => ({ fill: signedColor(v, max), strong: Math.abs(v) / max > 0.6 });

/* ------------------------------------------------------------------ */
/* Simulator                                                           */
/* ------------------------------------------------------------------ */

export default function ConvolutionSim() {
  const [n, setN] = useState(5);
  const [inputPreset, setInputPreset] = useState<InputPreset>("lesson");
  const [grid, setGrid] = useState<Grid>(() => makeInput("lesson", 5));
  const [kernelPreset, setKernelPreset] = useState<KernelPreset>("lesson");
  const [kernel, setKernel] = useState<Grid>(KERNELS.lesson.k);
  const [stride, setStride] = useState(1);
  const [padding, setPadding] = useState(0);
  const [bias, setBias] = useState(0);
  const [useRelu, setUseRelu] = useState(false);
  const [usePool, setUsePool] = useState(false);
  const [flip, setFlip] = useState(false);

  const k = 3;
  const effKernel = useMemo(() => (flip ? flipKernel(kernel) : kernel), [kernel, flip]);
  const xp = useMemo(() => pad2d(grid, padding), [grid, padding]);
  const Ho = outputSize(n, k, padding, stride);
  const Wo = Ho;
  const pre = useMemo(() => conv2d(grid, effKernel, { stride, padding, bias }), [grid, effKernel, stride, padding, bias]);
  const act = useMemo(() => (useRelu ? relu(pre) : pre), [pre, useRelu]);
  const pooled = useMemo(() => (usePool && Ho >= 2 ? pool2d(act, 2, 2, "max") : null), [usePool, Ho, act]);
  const Po = pooled ? pooled.length : 0;

  const convCount = Ho * Wo;
  const poolCount = Po * Po;
  const total = Math.max(1, convCount + poolCount);
  const s = useStepper(total, { baseMs: 900 });
  const step = Math.min(s.step, total - 1);
  const { setStep } = s;
  useEffect(() => {
    // keep the stepper inside range when stride/padding/size/pooling change the number of steps
    setStep((v) => Math.min(v, total - 1));
  }, [total, setStep]);
  const inPool = step >= convCount;
  const ci = inPool ? Ho - 1 : Math.floor(step / Wo);
  const cj = inPool ? Wo - 1 : step % Wo;
  const pIdx = inPool ? step - convCount : -1;
  const pi = pIdx >= 0 ? Math.floor(pIdx / Po) : -1;
  const pj = pIdx >= 0 ? pIdx % Po : -1;

  const cur = useMemo(() => convStep(xp, effKernel, ci, cj, stride, bias), [xp, effKernel, ci, cj, stride, bias]);
  const curOut = act[ci]?.[cj] ?? 0;

  const inMax = Math.max(9, maxAbs(grid));
  const kMax = maxAbs(effKernel);
  const outMax = maxAbs(act);

  /* ---- editing ---- */
  const chooseInput = (p: InputPreset) => {
    setInputPreset(p);
    const size = p === "lesson" ? 5 : Math.max(n, 7);
    setN(size);
    setGrid(makeInput(p, size));
    s.reset();
  };
  const chooseSize = (size: number) => {
    setN(size);
    setGrid(makeInput(inputPreset, size));
    s.reset();
  };
  const bump = (i: number, j: number, delta: number) => {
    setGrid((g) => g.map((r, a) => r.map((v, b) => (a === i && b === j ? (((Math.round(v) + delta) % 10) + 10) % 10 : v))));
  };
  const chooseKernel = (p: KernelPreset) => {
    setKernelPreset(p);
    if (p !== "custom") setKernel(KERNELS[p].k.map((r) => r.slice()));
  };
  const editKernel = (a: number, b: number, v: number) => {
    setKernelPreset("custom");
    setKernel((kk) => kk.map((r, i) => r.map((x, j) => (i === a && j === b ? (Number.isFinite(v) ? v : 0) : x))));
  };

  /* ---- derived narration ---- */
  const span = n + 2 * padding - k;
  const leftover = span % stride;
  const padCellsInWindow = (() => {
    let c = 0;
    for (let a = 0; a < k; a++)
      for (let b = 0; b < k; b++) {
        const y = cur.r + a - padding;
        const x = cur.c + b - padding;
        if (y < 0 || x < 0 || y >= n || x >= n) c++;
      }
    return c;
  })();

  const poolWindow = pooled && pi >= 0 ? [act[2 * pi][2 * pj], act[2 * pi][2 * pj + 1], act[2 * pi + 1][2 * pj], act[2 * pi + 1][2 * pj + 1]] : [];

  const kernelNote = kernelPreset === "custom" ? "Your own kernel. Try making the weights sum to 0 (an edge/difference detector) or to 1 (a brightness-preserving filter)." : KERNELS[kernelPreset].note;

  const selCls = "field py-1 text-xs";

  const settings = (
    <div className="grid gap-3 text-xs sm:grid-cols-2">
      <div className="flex flex-col gap-2">
        <label className="flex items-center gap-2">
          <span className="w-24 shrink-0 text-muted">Input</span>
          <select className={`${selCls} min-w-0 flex-1`} value={inputPreset} onChange={(e) => chooseInput(e.target.value as InputPreset)} aria-label="Input pattern">
            {INPUT_PRESETS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="w-24 shrink-0 text-muted">Input size n</span>
          <select className={`${selCls} min-w-0 flex-1`} value={n} onChange={(e) => chooseSize(+e.target.value)} aria-label="Input size">
            {[5, 6, 7, 8, 9, 10].map((v) => (
              <option key={v} value={v}>
                {v} × {v}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="w-24 shrink-0 text-muted">Kernel</span>
          <select className={`${selCls} min-w-0 flex-1`} value={kernelPreset} onChange={(e) => chooseKernel(e.target.value as KernelPreset)} aria-label="Kernel preset">
            {(Object.keys(KERNELS) as Exclude<KernelPreset, "custom">[]).map((id) => (
              <option key={id} value={id}>
                {KERNELS[id].label}
              </option>
            ))}
            <option value="custom">Custom (edit the cells)</option>
          </select>
        </label>
      </div>
      <div className="flex flex-col gap-2">
        <Slider label="Stride s" value={stride} min={1} max={3} step={1} onChange={(v) => setStride(v)} fmt={(v) => String(v)} />
        <Slider label="Padding p" value={padding} min={0} max={2} step={1} onChange={(v) => setPadding(v)} fmt={(v) => String(v)} />
        <Slider label="Bias b" value={bias} min={-5} max={5} step={0.5} onChange={setBias} fmt={(v) => nf(v)} />
        <div className="flex flex-wrap gap-x-4 gap-y-1">
          <label className="flex items-center gap-1.5">
            <input type="checkbox" checked={useRelu} onChange={(e) => setUseRelu(e.target.checked)} /> ReLU
          </label>
          <label className="flex items-center gap-1.5">
            <input type="checkbox" checked={usePool} onChange={(e) => setUsePool(e.target.checked)} /> 2×2 max-pool
          </label>
          <label className="flex items-center gap-1.5" title="Rotate the kernel by 180° before sliding: the textbook (signal-processing) convolution">
            <input type="checkbox" checked={flip} onChange={(e) => setFlip(e.target.checked)} /> Flip kernel (true convolution)
          </label>
        </div>
      </div>
    </div>
  );

  const prodTerms = cur.window.flatMap((row, a) => row.map((v, b) => `${nf(v)}·${nf(effKernel[a][b])}`));

  const aside = (
    <div className="flex flex-col gap-3">
      <div>
        <div className="mb-1 text-xs font-medium uppercase tracking-wide text-faint">Output size</div>
        <div className="rounded-md border border-border bg-surface-2/60 p-2 font-mono text-xs leading-relaxed">
          ⌊(n + 2p − k) / s⌋ + 1
          <br />= ⌊({n} + 2·{padding} − {k}) / {stride}⌋ + 1
          <br />= ⌊{span} / {stride}⌋ + 1 = <span className="font-semibold text-accent">{Ho}</span>
          <br />
          feature map: {Ho} × {Wo}
          {pooled && (
            <>
              <br />
              after 2×2 pool: ⌊({Ho} − 2) / 2⌋ + 1 = {Po}
            </>
          )}
        </div>
      </div>

      {!inPool ? (
        <div>
          <div className="mb-1 text-xs font-medium uppercase tracking-wide text-faint">
            Output [{ci}, {cj}] — 9 multiply-adds
          </div>
          <div className="rounded-md border border-border p-2 font-mono text-[11px] leading-relaxed">
            <div className="grid grid-cols-3 gap-1">
              {cur.products.flat().map((p, idx) => (
                <div key={idx} className="rounded px-1 py-0.5 text-center" style={{ background: signedColor(p, Math.max(1e-9, ...cur.products.flat().map(Math.abs))) }} title={prodTerms[idx]}>
                  <div className="text-faint">{prodTerms[idx]}</div>
                  <div className="font-semibold">{nf(p)}</div>
                </div>
              ))}
            </div>
            <div className="mt-2 break-words">
              Σ = {cur.products.flat().map((p) => nf(p)).join(" + ").replace(/\+ −/g, "− ")} = <b>{nf(cur.sum)}</b>
            </div>
            <div>
              {nf(cur.sum)} + b({nf(bias)}) = <b>{nf(cur.value)}</b>
              {useRelu && (
                <>
                  {" "}
                  → ReLU → <b className="text-accent">{nf(curOut)}</b>
                </>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div>
          <div className="mb-1 text-xs font-medium uppercase tracking-wide text-faint">
            Pool output [{pi}, {pj}]
          </div>
          <div className="rounded-md border border-border p-2 font-mono text-xs">
            max({poolWindow.map(nf).join(", ")}) = <b className="text-accent">{nf(pooled?.[pi]?.[pj] ?? 0)}</b>
          </div>
        </div>
      )}

      <Narration title={inPool ? `Max-pool step ${pIdx + 1} of ${poolCount}` : `Convolution step ${step + 1} of ${convCount}`}>
        {!inPool ? (
          <>
            The kernel’s top-left corner sits at padded row {cur.r}, column {cur.c}
            {step > 0 && cj > 0 ? ` — it moved ${stride} cell${stride > 1 ? "s" : ""} right` : step > 0 ? ` — it jumped back to the left edge and ${stride} row${stride > 1 ? "s" : ""} down` : ""}. Each of the 9 window values is multiplied by the kernel weight on top of it and the products are added
            {bias !== 0 ? ", then the bias is added" : ""}: the result {nf(cur.value)} becomes output cell [{ci}, {cj}]
            {useRelu && cur.value < 0 ? ", which ReLU clips to 0" : ""}.
            {padCellsInWindow > 0 && ` ${padCellsInWindow} of the 9 window cells are padding zeros, so they contribute nothing.`}
            {step === convCount - 1 && leftover > 0 && ` Note: (n + 2p − k) = ${span} is not divisible by s = ${stride}, so the last ${leftover} row(s)/column(s) of the padded input are never covered — that is the floor in the formula.`}
            {step === convCount - 1 && !pooled && " The feature map is complete."}
            <span className="mt-1 block text-xs">{kernelNote}</span>
          </>
        ) : (
          <>
            A 2×2 window with stride 2 slides over the {useRelu ? "ReLU-activated " : ""}feature map and keeps only the largest value. Each pooled cell summarises a non-overlapping 2×2 block, halving height and width ({Ho} → {Po}).
            {Ho % 2 === 1 && pIdx === poolCount - 1 && ` Because ${Ho} is odd, the last row and column of the feature map are dropped (floor).`}
          </>
        )}
      </Narration>
    </div>
  );

  const flipNote = flip ? "Kernel shown rotated 180° — this is what actually slides." : "Kernel slides as shown (cross-correlation, like PyTorch).";

  return (
    <SimShell
      title="Convolution Operation"
      subtitle="A 3×3 kernel slides over the (padded) input; every output cell is a 9-term multiply-add."
      controls={
        <Controls
          s={s}
          extra={
            <button
              className="sim-btn"
              onClick={() => {
                s.setPlaying(false);
                s.setStep(Math.max(0, convCount - 1));
              }}
              aria-label="Show the full feature map"
            >
              Fill map
            </button>
          }
        />
      }
      aside={aside}
    >
      {settings}
      <div className="mt-4 flex flex-wrap items-start gap-x-6 gap-y-4">
        <figure className="min-w-0 flex-[1_1_240px]" style={{ maxWidth: 380 }}>
          <figcaption className="mb-1 text-xs text-muted">
            Input {n}×{n}
            {padding > 0 ? ` + padding ${padding} → ${n + 2 * padding}×${n + 2 * padding}` : ""}
          </figcaption>
          <GridSvg
            values={xp}
            pad={padding}
            label={`Padded input grid, ${n + 2 * padding} by ${n + 2 * padding}, with the kernel window highlighted`}
            color={seqCell(inMax)}
            highlight={!inPool ? { r: cur.r, c: cur.c, h: k, w: k } : null}
            onCell={(i, j, e) => (e.shiftKey ? bump(i, j, -Math.round(grid[i][j])) : bump(i, j, 1))}
            onCellAlt={(i, j) => bump(i, j, -1)}
            maxWidth={380}
          />
          <p className="mt-1 text-[11px] text-faint">Click a cell: +1 · right-click: −1 · shift-click: 0. Dashed cells are zero padding.</p>
        </figure>

        <div className="flex min-w-0 flex-[0_1_150px] flex-col gap-1">
          <div className="text-xs text-muted">Kernel 3×3 (editable)</div>
          <div className="grid grid-cols-3 gap-1" style={{ width: 150 }}>
            {kernel.map((row, a) =>
              row.map((v, b) => {
                const shownV = flip ? effKernel[a][b] : v;
                return (
                  <NumCell
                    key={`${a}-${b}`}
                    value={shownV}
                    onCommit={(nv) => {
                      // when flipped, edit the underlying (unflipped) cell
                      const [ra, rb] = flip ? [2 - a, 2 - b] : [a, b];
                      editKernel(ra, rb, nv);
                    }}
                    background={signedColor(shownV, kMax)}
                    label={`Kernel weight row ${a + 1} column ${b + 1}`}
                  />
                );
              }),
            )}
          </div>
          <div className="text-[11px] text-faint" style={{ maxWidth: 150 }}>
            {flipNote} Bias b = {nf(bias)}.
          </div>
        </div>

        <figure className="min-w-0 flex-[1_1_160px]" style={{ maxWidth: 260 }}>
          <figcaption className="mb-1 text-xs text-muted">
            Feature map {Ho}×{Wo}
            {useRelu ? " (after ReLU)" : ""}
          </figcaption>
          <GridSvg
            values={act}
            label={`Output feature map ${Ho} by ${Wo}`}
            cs={30}
            color={signedCell(outMax)}
            shown={(i, j) => inPool || i * Wo + j <= step}
            highlight={inPool && pi >= 0 ? { r: 2 * pi, c: 2 * pj, h: 2, w: 2 } : { r: ci, c: cj, h: 1, w: 1 }}
            onCell={(i, j) => {
              s.setPlaying(false);
              s.setStep(i * Wo + j);
            }}
            maxWidth={260}
          />
          <p className="mt-1 text-[11px] text-faint">Click an output cell to jump to it.</p>
        </figure>

        {pooled && (
          <figure className="min-w-0 flex-[0_1_120px]" style={{ maxWidth: 160 }}>
            <figcaption className="mb-1 text-xs text-muted">
              Max-pooled {Po}×{Po}
            </figcaption>
            <GridSvg
              values={pooled}
              label={`Max-pooled map ${Po} by ${Po}`}
              cs={30}
              color={signedCell(outMax)}
              shown={(i, j) => inPool && i * Po + j <= pIdx}
              highlight={inPool && pi >= 0 ? { r: pi, c: pj, h: 1, w: 1 } : null}
              maxWidth={160}
            />
          </figure>
        )}
      </div>
    </SimShell>
  );
}
