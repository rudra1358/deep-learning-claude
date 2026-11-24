"use client";

import { useMemo, useState, type ReactNode } from "react";
import { Controls, Narration, SimShell, Slider, signedColor, useStepper } from "./SimShell";
import { pad2d, type Grid } from "@/lib/sim/conv";
import { mulberry32 } from "@/lib/sim/random";
import {
  convCosts,
  depthwise,
  dilatedConv,
  dilatedOutputSize,
  dilatedStep,
  effectiveKernel,
  pointwise,
  stackReceptiveField,
  transposedConv,
  transposedOutputSize,
} from "@/lib/sim/convVariants";

/* ------------------------------------------------------------------ */
/* Shared helpers                                                      */
/* ------------------------------------------------------------------ */

function nf(v: number): string {
  const r = Math.round(v * 100) / 100;
  const s = Number.isInteger(r) ? String(r) : Math.abs(r) >= 10 ? r.toFixed(1) : r.toFixed(2);
  return s.replace("-", "−");
}

const big = (v: number) => v.toLocaleString("en-US");

const maxAbs = (g: Grid) => g.reduce((m, r) => r.reduce((mm, v) => Math.max(mm, Math.abs(v)), m), 0) || 1;

function randInts(h: number, w: number, seed: number, lo: number, hi: number): Grid {
  const rand = mulberry32(seed);
  return Array.from({ length: h }, () => Array.from({ length: w }, () => lo + Math.floor(rand() * (hi - lo + 1))));
}

type CellStyle = { fill?: string; stroke?: string; strokeWidth?: number; dashed?: boolean; text?: string; textColor?: string; sub?: string };

/** Generic SVG grid of cells; `cell` decides how each one looks. */
function CellGrid({ h, w, cs, label, cell, maxWidth, overlay }: { h: number; w: number; cs: number; label: string; cell: (i: number, j: number) => CellStyle; maxWidth: number; overlay?: ReactNode }) {
  const fs = Math.max(8, Math.min(13, cs * 0.38));
  return (
    <svg viewBox={`-3 -3 ${w * cs + 6} ${h * cs + 6}`} className="h-auto w-full" style={{ maxWidth }} role="img" aria-label={label}>
      {Array.from({ length: h }, (_, i) =>
        Array.from({ length: w }, (_, j) => {
          const st = cell(i, j);
          return (
            <g key={`${i}-${j}`}>
              <rect
                x={j * cs}
                y={i * cs}
                width={cs}
                height={cs}
                fill={st.fill ?? "var(--surface)"}
                stroke={st.stroke ?? "var(--border)"}
                strokeWidth={st.strokeWidth ?? 1}
                strokeDasharray={st.dashed ? "3 2" : undefined}
              />
              {st.text !== undefined && (
                <text x={j * cs + cs / 2} y={i * cs + cs / 2 + fs * 0.36} textAnchor="middle" fontSize={fs} fontFamily="var(--font-mono, monospace)" fill={st.textColor ?? "var(--text)"} style={{ pointerEvents: "none" }}>
                  {st.text}
                </text>
              )}
              {st.sub !== undefined && (
                <text x={j * cs + cs - 2} y={i * cs + fs * 0.75} textAnchor="end" fontSize={fs * 0.62} fontFamily="var(--font-mono, monospace)" fill="var(--text-muted)" style={{ pointerEvents: "none" }}>
                  {st.sub}
                </text>
              )}
            </g>
          );
        }),
      )}
      {overlay}
    </svg>
  );
}

function Seg<T extends string | number>({ label, value, options, onChange }: { label: string; value: T; options: { v: T; label: string }[]; onChange: (v: T) => void }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs">
      <span className="w-24 shrink-0 text-muted">{label}</span>
      {options.map((o) => (
        <button key={String(o.v)} className={`sim-btn ${o.v === value ? "primary" : ""}`} aria-pressed={o.v === value} onClick={() => onChange(o.v)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-3 border-b border-border py-1 last:border-b-0">
      <span className="text-muted">{label}</span>
      <span className="font-mono">{value}</span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* (a) Dilated                                                         */
/* ------------------------------------------------------------------ */

const DIL_KERNELS: Record<string, { label: string; k: Grid }> = {
  sobel: { label: "Sobel-x", k: [[1, 0, -1], [2, 0, -2], [1, 0, -1]] },
  lap: { label: "Laplacian", k: [[0, 1, 0], [1, -4, 1], [0, 1, 0]] },
  box: { label: "Box (all 1)", k: [[1, 1, 1], [1, 1, 1], [1, 1, 1]] },
};

function DilatedMode() {
  const [n, setN] = useState(9);
  const [d, setD] = useState(2);
  const [same, setSame] = useState(false);
  const [kname, setKname] = useState("sobel");
  const [seed, setSeed] = useState(1);
  const kernel = DIL_KERNELS[kname].k;
  const k = 3;
  const p = same ? d : 0; // "same" padding for a 3×3 kernel with dilation d is p = d
  const x = useMemo(() => randInts(n, n, seed, 0, 4), [n, seed]);
  const xp = useMemo(() => pad2d(x, p), [x, p]);
  const outN = dilatedOutputSize(n, k, p, 1, d);
  const Y = useMemo(() => (outN > 0 ? dilatedConv(x, kernel, d, p, 1) : []), [x, kernel, d, p, outN]);
  const total = Math.max(1, outN * outN);
  const s = useStepper(total, { baseMs: 1100 });
  const step = Math.min(s.step, total - 1);
  const i = outN > 0 ? Math.floor(step / outN) : 0;
  const j = outN > 0 ? step % outN : 0;
  const st = outN > 0 ? dilatedStep(xp, kernel, i, j, d, 1) : null;
  const tapSet = new Map<string, number>();
  st?.taps.flat().forEach((t) => tapSet.set(`${t.r},${t.c}`, t.w));
  const kEff = effectiveKernel(k, d);
  const P = xp.length;
  const cs = 30;
  const mY = Y.length ? maxAbs(Y) : 1;
  const doubling = [1, 2, 4];

  return (
    <SimShell
      title="Dilated (atrous) convolution"
      subtitle="A 3×3 kernel whose taps are spread d pixels apart: same 9 weights, wider view."
      controls={
        <div className="space-y-2">
          <Controls s={s} />
          <div className="grid gap-2 sm:grid-cols-2">
            <Slider label="Dilation d" value={d} min={1} max={4} step={1} onChange={(v) => { setD(v); s.reset(); }} fmt={(v) => String(v)} />
            <Slider label="Input size" value={n} min={7} max={13} step={1} onChange={(v) => { setN(v); s.reset(); }} fmt={(v) => `${v}×${v}`} />
          </div>
          <div className="flex flex-wrap gap-3">
            <Seg label="Kernel" value={kname} options={Object.entries(DIL_KERNELS).map(([v, o]) => ({ v, label: o.label }))} onChange={(v) => setKname(v)} />
            <Seg label="Padding" value={same ? "same" : "valid"} options={[{ v: "valid", label: "valid (p=0)" }, { v: "same", label: `same (p=d=${d})` }]} onChange={(v) => { setSame(v === "same"); s.reset(); }} />
            <button className="sim-btn" onClick={() => { setSeed((v) => v + 1); s.reset(); }}>New input</button>
          </div>
        </div>
      }
      aside={
        <div className="space-y-3">
          <div>
            <div className="mb-1 font-medium">Readout</div>
            <Stat label="effective kernel k + (k−1)(d−1)" value={`${k} + ${k - 1}·${d - 1} = ${kEff}`} />
            <Stat label="padding p" value={String(p)} />
            <Stat label="output size" value={outN > 0 ? `⌊(${n} + ${2 * p} − ${kEff})/1⌋ + 1 = ${outN}` : "kernel does not fit"} />
            <Stat label="weights" value="9 (for every d)" />
            <Stat label="RF of this layer" value={`${kEff}×${kEff}`} />
            <Stat label="RF of 3 layers, all d" value={`${stackReceptiveField(k, [d, d, d])}`} />
            <Stat label="RF of 3 layers, d = 1,2,4" value={`${stackReceptiveField(k, doubling)}`} />
          </div>
          <div className="text-xs text-muted">Kernel ({DIL_KERNELS[kname].label}): {kernel.map((r) => `[${r.map(nf).join(" ")}]`).join(" ")}</div>
        </div>
      }
    >
      <div className="flex flex-wrap items-start gap-4">
        <div className="min-w-0 flex-1" style={{ minWidth: 200 }}>
          <div className="mb-1 text-xs text-muted">Input {n}×{n}{p ? ` + ${p} rings of zero padding` : ""} — gold cells are the 9 sampled pixels</div>
          <CellGrid
            h={P}
            w={P}
            cs={cs}
            maxWidth={420}
            label="Padded input with the dilated kernel taps highlighted"
            cell={(r, c) => {
              const isPad = r < p || c < p || r >= P - p || c >= P - p;
              const tw = tapSet.get(`${r},${c}`);
              const inSpan = st && r >= st.r0 && r < st.r0 + kEff && c >= st.c0 && c < st.c0 + kEff;
              return {
                fill: tw !== undefined ? "color-mix(in srgb, var(--gold) 45%, var(--surface))" : isPad ? "var(--surface-2)" : inSpan ? "color-mix(in srgb, var(--gold) 10%, var(--surface))" : "var(--surface)",
                stroke: tw !== undefined ? "var(--gold)" : undefined,
                strokeWidth: tw !== undefined ? 2.5 : 1,
                dashed: isPad,
                text: nf(xp[r][c]),
                textColor: isPad ? "var(--text-muted)" : "var(--text)",
                sub: tw !== undefined ? `×${nf(tw)}` : undefined,
              };
            }}
            overlay={st ? <rect x={st.c0 * cs} y={st.r0 * cs} width={kEff * cs} height={kEff * cs} fill="none" stroke="var(--gold)" strokeDasharray="5 3" strokeWidth={1.5} /> : null}
          />
        </div>
        <div className="min-w-0" style={{ width: 200 }}>
          <div className="mb-1 text-xs text-muted">Output {outN}×{outN}</div>
          {outN > 0 ? (
            <CellGrid
              h={outN}
              w={outN}
              cs={30}
              maxWidth={200}
              label="Output feature map"
              cell={(a, b) => {
                const idx = a * outN + b;
                const shown = idx <= step;
                return {
                  fill: shown ? signedColor(Y[a][b], mY) : "var(--surface)",
                  stroke: idx === step ? "var(--gold)" : undefined,
                  strokeWidth: idx === step ? 3 : 1,
                  text: shown ? nf(Y[a][b]) : "",
                };
              }}
            />
          ) : (
            <div className="text-sm text-clay">The dilated kernel spans {kEff} pixels but the padded input is only {n + 2 * p} wide. Use same padding or a larger input.</div>
          )}
        </div>
      </div>
      <div className="mt-3">
        <Narration title={st ? `Output (${i}, ${j}) — taps every ${d} pixel${d > 1 ? "s" : ""}` : "No valid position"}>
          {st ? (
            <>
              The window starts at padded row {st.r0}, column {st.c0} and spans {kEff}×{kEff} pixels, but only the 9 gold pixels (rows {st.taps.map((r) => r[0].r).join(", ")}; columns {st.taps[0].map((t) => t.c).join(", ")}) are read.{" "}
              <span className="font-mono">
                {st.taps.flat().map((t, q) => `${q ? " + " : ""}${nf(t.v)}·${nf(t.w)}`).join("")} = {nf(st.sum)}
              </span>
              .{" "}
              {d > 1
                ? `The ${kEff * kEff - 9} skipped pixels inside the dashed square never influence this output — stacking layers with the same d leaves the same holes (gridding).`
                : "With d = 1 this is an ordinary 3×3 convolution."}
            </>
          ) : (
            "Change the settings so that the kernel fits."
          )}
        </Narration>
      </div>
    </SimShell>
  );
}

/* ------------------------------------------------------------------ */
/* (b) Transposed                                                      */
/* ------------------------------------------------------------------ */

function TransposedMode() {
  const [n, setN] = useState(3);
  const [k, setK] = useState(3);
  const [stride, setStride] = useState(2);
  const [p, setP] = useState(0);
  const [op, setOp] = useState(0);
  const [kind, setKind] = useState<"ones" | "random">("ones");
  const [view, setView] = useState<"values" | "count">("values");
  const [seed, setSeed] = useState(3);
  const opC = Math.min(op, stride - 1);
  const pC = Math.min(p, k - 1);
  const x = useMemo(() => (kind === "ones" ? randInts(n, n, 0, 1, 1) : randInts(n, n, seed, 1, 3)), [n, kind, seed]);
  const kernel = useMemo(() => (kind === "ones" ? randInts(k, k, 0, 1, 1) : randInts(k, k, seed + 100, -1, 2)), [k, kind, seed]);
  const total = n * n + 1;
  const s = useStepper(total, { baseMs: 1200 });
  const stamps = Math.min(s.step + 1, n * n);
  const finalStep = s.step >= n * n;
  const state = useMemo(() => transposedConv(x, kernel, stride, pC, opC, stamps), [x, kernel, stride, pC, opC, stamps]);
  const finalState = useMemo(() => transposedConv(x, kernel, stride, pC, opC), [x, kernel, stride, pC, opC]);
  const ci = Math.floor((stamps - 1) / n);
  const cj = (stamps - 1) % n;
  const full = state.full;
  const outSize = transposedOutputSize(n, k, stride, pC, opC);
  const mC = maxAbs(finalState.canvas);
  const maxCount = maxAbs(finalState.count);
  const cs = full > 10 ? 26 : 32;
  const counts = new Set(finalState.out.flat().length ? finalState.count.slice(pC, full - pC).flatMap((r) => r.slice(pC, full - pC)) : []);
  const uneven = counts.size > 1;

  return (
    <SimShell
      title="Transposed convolution"
      subtitle="Each input value stamps a scaled copy of the kernel onto the output canvas; overlapping stamps add up."
      controls={
        <div className="space-y-2">
          <Controls s={s} />
          <div className="grid gap-2 sm:grid-cols-2">
            <Slider label="Input size" value={n} min={2} max={4} step={1} onChange={(v) => { setN(v); s.reset(); }} fmt={(v) => `${v}×${v}`} />
            <Slider label="Kernel k" value={k} min={2} max={4} step={1} onChange={(v) => { setK(v); s.reset(); }} fmt={(v) => `${v}×${v}`} />
            <Slider label="Stride s" value={stride} min={1} max={3} step={1} onChange={(v) => { setStride(v); s.reset(); }} fmt={(v) => String(v)} />
            <Slider label="Padding p" value={pC} min={0} max={k - 1} step={1} onChange={(v) => setP(v)} fmt={(v) => String(v)} />
            <Slider label="output_padding" value={opC} min={0} max={Math.max(0, stride - 1)} step={1} onChange={(v) => setOp(v)} fmt={(v) => String(v)} />
          </div>
          <div className="flex flex-wrap gap-3">
            <Seg label="Values" value={kind} options={[{ v: "ones", label: "all ones" }, { v: "random", label: "random ints" }]} onChange={(v) => { setKind(v); s.reset(); }} />
            <Seg label="Show" value={view} options={[{ v: "values", label: "sums" }, { v: "count", label: "overlap count" }]} onChange={setView} />
            {kind === "random" && <button className="sim-btn" onClick={() => { setSeed((v) => v + 1); s.reset(); }}>New values</button>}
          </div>
        </div>
      }
      aside={
        <div className="space-y-3">
          <div className="mb-1 font-medium">Readout</div>
          <Stat label="canvas (n−1)s + k + op" value={`${(n - 1)}·${stride} + ${k} + ${opC} = ${full}`} />
          <Stat label="output (n−1)s − 2p + k + op" value={`${outSize}×${outSize}`} />
          <Stat label="stamps done" value={`${stamps}/${n * n}`} />
          <Stat label="overlap counts in output" value={[...counts].sort((a, b) => a - b).join(", ") || "—"} />
          <div className="text-xs text-muted">
            Input: {x.map((r) => `[${r.join(" ")}]`).join(" ")}
            <br />
            Kernel: {kernel.map((r) => `[${r.map(nf).join(" ")}]`).join(" ")}
          </div>
          {uneven ? (
            <div className="text-xs text-clay">k = {k} is not divisible by s = {stride}: output pixels receive different numbers of stamps — the checkerboard pattern.</div>
          ) : (
            <div className="text-xs text-muted">Every output pixel receives the same number of stamps: no checkerboard from overlap.</div>
          )}
        </div>
      }
    >
      <div className="flex flex-wrap items-start gap-4">
        <div style={{ width: 130 }} className="min-w-0">
          <div className="mb-1 text-xs text-muted">Input {n}×{n}</div>
          <CellGrid
            h={n}
            w={n}
            cs={34}
            maxWidth={130}
            label="Input grid; the current pixel is highlighted"
            cell={(a, b) => {
              const idx = a * n + b;
              const cur = !finalStep && a === ci && b === cj;
              return {
                fill: idx < stamps ? "color-mix(in srgb, var(--accent) 25%, var(--surface))" : "var(--surface)",
                stroke: cur ? "var(--gold)" : undefined,
                strokeWidth: cur ? 3 : 1,
                text: nf(x[a][b]),
              };
            }}
          />
        </div>
        <div className="min-w-0 flex-1" style={{ minWidth: 220 }}>
          <div className="mb-1 text-xs text-muted">
            Canvas {full}×{full}
            {pC > 0 ? ` — dashed border rows/cols are cropped by p = ${pC}` : ""}
          </div>
          <CellGrid
            h={full}
            w={full}
            cs={cs}
            maxWidth={440}
            label="Output canvas with accumulated stamps"
            cell={(r, c) => {
              const cropped = r < pC || c < pC || r >= full - pC || c >= full - pC;
              const inStamp = !finalStep && r >= ci * stride && r < ci * stride + k && c >= cj * stride && c < cj * stride + k;
              const v = state.canvas[r][c];
              const cnt = state.count[r][c];
              const fill =
                view === "count"
                  ? cnt
                    ? `color-mix(in srgb, var(--accent) ${Math.round(15 + (85 * cnt) / maxCount)}%, transparent)`
                    : "var(--surface)"
                  : cnt
                    ? signedColor(v, mC)
                    : "var(--surface)";
              return {
                fill: finalStep && cropped ? "var(--surface-2)" : fill,
                stroke: inStamp ? "var(--gold)" : undefined,
                strokeWidth: inStamp ? 2.5 : 1,
                dashed: cropped,
                text: cnt ? (view === "count" ? String(cnt) : nf(v)) : "",
                textColor: finalStep && cropped ? "var(--text-muted)" : "var(--text)",
                sub: view === "values" && cnt > 1 ? `${cnt}` : undefined,
              };
            }}
          />
        </div>
      </div>
      <div className="mt-3">
        <Narration title={finalStep ? `Final output ${outSize}×${outSize}` : `Stamp ${stamps}: input (${ci}, ${cj}) = ${nf(x[ci][cj])}`}>
          {finalStep ? (
            <>
              All {n * n} stamps are placed. Cropping p = {pC} from each side of the {full}×{full} canvas leaves the {outSize}×{outSize} output
              {opC ? ` (output_padding = ${opC} added one extra row and column at the bottom/right before cropping)` : ""}. Switch to <b>overlap count</b>: with all-ones inputs and kernel, each output value equals the number of stamps that covered it
              {uneven ? " — the uneven counts are exactly the checkerboard artifact." : "."}
            </>
          ) : (
            <>
              Pixel ({ci}, {cj}) lands at canvas position ({ci}·{stride}, {cj}·{stride}) = ({ci * stride}, {cj * stride}) and adds {nf(x[ci][cj])} × K to the gold {k}×{k} block. Small numbers in the corner show how many stamps have
              overlapped there so far. Stride {stride} moves the stamp {stride} canvas cells per input pixel, so the output is about {stride}× larger than the input.
            </>
          )}
        </Narration>
      </div>
    </SimShell>
  );
}

/* ------------------------------------------------------------------ */
/* (c) Depthwise-separable                                             */
/* ------------------------------------------------------------------ */

const LOG_CHOICES = [1, 3, 8, 16, 32, 64, 128, 256, 512, 1024];

function SeparableMode() {
  const [cin, setCin] = useState(3);
  const [cout, setCout] = useState(4);
  const [seed, setSeed] = useState(7);
  const N = 5;
  const x = useMemo(() => Array.from({ length: cin }, (_, c) => randInts(N, N, seed * 31 + c, 0, 4)), [cin, seed]);
  const dk = useMemo(() => Array.from({ length: cin }, (_, c) => randInts(3, 3, seed * 57 + c + 11, -1, 1)), [cin, seed]);
  const P = useMemo(() => randInts(cout, cin, seed * 13 + 5, -2, 2), [cout, cin, seed]);
  const D = useMemo(() => depthwise(x, dk), [x, dk]);
  const Y = useMemo(() => pointwise(D, P), [D, P]);

  const total = 1 + cin + cout;
  const s = useStepper(total, { baseMs: 1300 });
  const step = Math.min(s.step, total - 1);
  const phase = step === 0 ? "input" : step <= cin ? "dw" : "pw";
  const curC = phase === "dw" ? step - 1 : -1;
  const curO = phase === "pw" ? step - 1 - cin : -1;

  // calculator
  const [ci2, setCi2] = useState(4); // index into LOG_CHOICES -> 32
  const [co2, setCo2] = useState(5); // 64
  const [k2, setK2] = useState(3);
  const [hw, setHw] = useState(56);
  const C = convCosts(LOG_CHOICES[ci2], LOG_CHOICES[co2], k2, hw, hw);

  // diagram geometry
  const tile = 50;
  const gap = 12;
  const rows = Math.max(cin, cout);
  const Hsvg = rows * (tile + gap) + 30;
  const colX = [10, 170, 330];
  const yOf = (idx: number, count: number) => 24 + idx * (tile + gap) + ((rows - count) * (tile + gap)) / 2;
  const mx = Math.max(...x.map(maxAbs));
  const mD = Math.max(...D.map(maxAbs));
  const mY = Math.max(...Y.map(maxAbs));
  const mid = 2;

  function Tile({ g, m, X0, Y0, active, label }: { g: Grid; m: number; X0: number; Y0: number; active: boolean; label: string }) {
    const c = tile / N;
    return (
      <g aria-label={label}>
        {g.map((row, i) => row.map((v, j) => <rect key={`${i}-${j}`} x={X0 + j * c} y={Y0 + i * c} width={c} height={c} fill={signedColor(v, m)} stroke="var(--border)" strokeWidth={0.4} />))}
        <rect x={X0} y={Y0} width={tile} height={tile} fill="none" stroke={active ? "var(--gold)" : "var(--border)"} strokeWidth={active ? 3 : 1} />
        <rect x={X0 + mid * c} y={Y0 + mid * c} width={c} height={c} fill="none" stroke="var(--text)" strokeWidth={0.8} />
      </g>
    );
  }

  const dwCenter = curC >= 0 ? D[curC][mid][mid] : 0;
  const pwTerms = curO >= 0 ? P[curO].map((w, c) => ({ w, v: D[c][mid][mid] })) : [];

  return (
    <SimShell
      title="Depthwise-separable convolution"
      subtitle="Depthwise: one 3×3 filter per channel (no mixing). Pointwise: a 1×1 convolution mixes channels at every pixel."
      controls={
        <div className="space-y-2">
          <Controls s={s} />
          <div className="grid gap-2 sm:grid-cols-2">
            <Slider label="Demo C_in" value={cin} min={1} max={4} step={1} onChange={(v) => { setCin(v); s.reset(); }} fmt={(v) => String(v)} />
            <Slider label="Demo C_out" value={cout} min={1} max={4} step={1} onChange={(v) => { setCout(v); s.reset(); }} fmt={(v) => String(v)} />
          </div>
          <button className="sim-btn" onClick={() => setSeed((v) => v + 1)}>New weights</button>
        </div>
      }
      aside={
        <div className="space-y-2">
          <div className="font-medium">Cost calculator (weights, stride 1, same padding)</div>
          <Slider label="C_in" value={ci2} min={0} max={LOG_CHOICES.length - 1} step={1} onChange={setCi2} fmt={(v) => String(LOG_CHOICES[v])} />
          <Slider label="C_out" value={co2} min={0} max={LOG_CHOICES.length - 1} step={1} onChange={setCo2} fmt={(v) => String(LOG_CHOICES[v])} />
          <Slider label="kernel k" value={k2} min={1} max={7} step={2} onChange={setK2} fmt={(v) => `${v}×${v}`} />
          <Slider label="H = W" value={hw} min={7} max={224} step={7} onChange={setHw} fmt={(v) => String(v)} />
          <div className="pt-1 text-xs">
            <Stat label="standard params" value={big(C.standardParams)} />
            <Stat label="depthwise params" value={big(C.depthwiseParams)} />
            <Stat label="pointwise params" value={big(C.pointwiseParams)} />
            <Stat label="separable params" value={big(C.separableParams)} />
            <Stat label="standard MACs" value={big(C.standardMacs)} />
            <Stat label="separable MACs" value={big(C.separableMacs)} />
            <Stat label="ratio separable/standard" value={C.ratio.toFixed(4)} />
            <Stat label="1/C_out + 1/k²" value={C.formulaRatio.toFixed(4)} />
            <Stat label="reduction factor" value={`${(1 / C.ratio).toFixed(2)}×`} />
            <Stat label="share of MACs in pointwise" value={`${((100 * C.pointwiseMacs) / C.separableMacs).toFixed(1)}%`} />
          </div>
        </div>
      }
    >
      <svg viewBox={`0 0 400 ${Hsvg}`} className="h-auto w-full" style={{ maxWidth: 560 }} role="img" aria-label="Channel stacks: input, depthwise outputs and pointwise outputs">
        <text x={colX[0]} y={14} fontSize={11} fill="var(--text-muted)">input ({cin} ch)</text>
        <text x={colX[1]} y={14} fontSize={11} fill="var(--text-muted)">depthwise ({cin} ch)</text>
        <text x={colX[2]} y={14} fontSize={11} fill="var(--text-muted)">pointwise ({cout} ch)</text>
        {/* depthwise arrows: channel c -> channel c only */}
        {x.map((_, c) => (
          <g key={`dwline${c}`}>
            <line x1={colX[0] + tile + 2} y1={yOf(c, cin) + tile / 2} x2={colX[1] - 2} y2={yOf(c, cin) + tile / 2} stroke={curC === c ? "var(--gold)" : "var(--border)"} strokeWidth={curC === c ? 2.5 : 1.2} />
            <text x={(colX[0] + tile + colX[1]) / 2} y={yOf(c, cin) + tile / 2 - 4} fontSize={9} textAnchor="middle" fill="var(--text-muted)">3×3 K{c}</text>
          </g>
        ))}
        {/* pointwise lines: every depthwise channel -> every output */}
        {P.map((row, o) =>
          row.map((w, c) => (
            <line
              key={`pw${o}-${c}`}
              x1={colX[1] + tile + 2}
              y1={yOf(c, cin) + tile / 2}
              x2={colX[2] - 2}
              y2={yOf(o, cout) + tile / 2}
              stroke={w === 0 ? "var(--border)" : w > 0 ? "var(--accent)" : "var(--clay)"}
              strokeOpacity={curO === -1 || curO === o ? 0.9 : 0.15}
              strokeWidth={curO === o ? 1 + Math.abs(w) : 0.6 + Math.abs(w) * 0.4}
            />
          )),
        )}
        {x.map((g, c) => <Tile key={`x${c}`} g={g} m={mx} X0={colX[0]} Y0={yOf(c, cin)} active={phase === "input" || curC === c} label={`input channel ${c}`} />)}
        {D.map((g, c) => (phase === "input" || (phase === "dw" && c > curC) ? null : <Tile key={`d${c}`} g={g} m={mD} X0={colX[1]} Y0={yOf(c, cin)} active={curC === c} label={`depthwise output ${c}`} />))}
        {Y.map((g, o) => (phase !== "pw" || o > curO ? null : <Tile key={`y${o}`} g={g} m={mY} X0={colX[2]} Y0={yOf(o, cout)} active={curO === o} label={`pointwise output ${o}`} />))}
      </svg>
      <div className="mt-3">
        <Narration title={phase === "input" ? "Input stack" : phase === "dw" ? `Depthwise: channel ${curC}` : `Pointwise: output channel ${curO}`}>
          {phase === "input" && (
            <>
              {cin} input channels of size {N}×{N}. A standard 3×3 layer to {cout} channels would need {cout}·{cin}·9 = {cout * cin * 9} weights; the separable pair needs {cin}·9 + {cin}·{cout} = {cin * 9 + cin * cout}. The outlined centre pixel is followed through both stages.
            </>
          )}
          {phase === "dw" && (
            <>
              Channel {curC} is correlated with its own kernel K{curC} = {dk[curC].map((r) => `[${r.map(nf).join(" ")}]`).join(" ")} (same padding) and with nothing else — no information crosses channels. Centre value:{" "}
              <span className="font-mono">
                {dk[curC].flatMap((r, a) => r.map((w, b) => `${nf(x[curC][mid - 1 + a][mid - 1 + b])}·${nf(w)}`)).join(" + ")} = {nf(dwCenter)}
              </span>
              .
            </>
          )}
          {phase === "pw" && (
            <>
              Output channel {curO} is a weighted sum of the {cin} depthwise maps, the same weights at every pixel (a 1×1 convolution = a {cin}→{cout} linear layer applied per pixel). Centre value:{" "}
              <span className="font-mono">
                {pwTerms.map((t, q) => `${q ? " + " : ""}${nf(t.w)}·${nf(t.v)}`).join("")} = {nf(Y[curO][mid][mid])}
              </span>
              . Line colour: teal = positive weight, clay = negative, thickness = |weight|.
            </>
          )}
        </Narration>
      </div>
    </SimShell>
  );
}

/* ------------------------------------------------------------------ */
/* Root                                                                */
/* ------------------------------------------------------------------ */

type Mode = "dilated" | "transposed" | "separable";

export default function ConvVariantsSim() {
  const [mode, setMode] = useState<Mode>("dilated");
  const tabs: { id: Mode; label: string }[] = [
    { id: "dilated", label: "Dilated" },
    { id: "transposed", label: "Transposed" },
    { id: "separable", label: "Depthwise-separable" },
  ];
  return (
    <div className="min-w-0">
      <div className="mb-[-0.75rem] mt-6 flex flex-wrap gap-2" role="tablist" aria-label="Convolution variant">
        {tabs.map((t) => (
          <button key={t.id} role="tab" aria-selected={mode === t.id} className={`sim-btn ${mode === t.id ? "primary" : ""}`} onClick={() => setMode(t.id)}>
            {t.label}
          </button>
        ))}
      </div>
      {mode === "dilated" && <DilatedMode />}
      {mode === "transposed" && <TransposedMode />}
      {mode === "separable" && <SeparableMode />}
    </div>
  );
}
