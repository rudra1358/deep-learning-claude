"use client";

import { useMemo, useRef, useState, type PointerEvent as RPointerEvent, type ReactNode, type RefObject } from "react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import {
  encode,
  iouParts,
  makeAnchors,
  makeCandidates,
  matchAnchors,
  nmsTrace,
  precisionRecall,
  toCxcywh,
  type Box,
  type Candidate,
} from "@/lib/sim/detection";

/* ------------------------------------------------------------------ */
/* Scene + shared helpers                                               */
/* ------------------------------------------------------------------ */

const W = 320;
const H = 256;
const MIN = 12;

const DEFAULT_GT: Box[] = [
  { x1: 36, y1: 52, x2: 140, y2: 180 },
  { x1: 170, y1: 36, x2: 282, y2: 108 },
  { x1: 196, y1: 140, x2: 262, y2: 226 },
];

const r1 = (v: number) => Math.round(v * 10) / 10;

function Stat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex justify-between gap-3 border-b border-border py-1 last:border-b-0">
      <span className="text-muted">{label}</span>
      <span className="font-mono">{value}</span>
    </div>
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

function Toggle({ label, values, options, onToggle }: { label: string; values: number[]; options: number[]; onToggle: (v: number) => void }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs">
      <span className="w-24 shrink-0 text-muted">{label}</span>
      {options.map((o) => {
        const on = values.includes(o);
        return (
          <button key={o} className={`sim-btn ${on ? "primary" : ""}`} aria-pressed={on} onClick={() => onToggle(o)}>
            {o}
          </button>
        );
      })}
    </div>
  );
}

const fmtBox = (b: Box) => `(${Math.round(b.x1)}, ${Math.round(b.y1)}, ${Math.round(b.x2)}, ${Math.round(b.y2)})`;

/**
 * An SVG canvas in scene units with draggable / resizable boxes.
 * Drag a box body to move it; drag its bottom-right square handle to resize.
 */
function useDrag(boxes: Box[], setBox: (i: number, b: Box) => void) {
  const svgRef = useRef<SVGSVGElement>(null);
  const drag = useRef<{ i: number; mode: "move" | "resize"; px: number; py: number; orig: Box } | null>(null);
  const toScene = (e: RPointerEvent) => {
    const svg = svgRef.current;
    if (!svg) return { x: 0, y: 0 };
    const r = svg.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * W, y: ((e.clientY - r.top) / r.height) * H };
  };
  const start = (i: number, mode: "move" | "resize") => (e: RPointerEvent) => {
    e.stopPropagation();
    e.preventDefault();
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
    const p = toScene(e);
    drag.current = { i, mode, px: p.x, py: p.y, orig: boxes[i] };
  };
  const move = (e: RPointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const p = toScene(e);
    const dx = p.x - d.px;
    const dy = p.y - d.py;
    const o = d.orig;
    if (d.mode === "move") {
      const w = o.x2 - o.x1;
      const h = o.y2 - o.y1;
      const x1 = Math.min(Math.max(0, o.x1 + dx), W - w);
      const y1 = Math.min(Math.max(0, o.y1 + dy), H - h);
      setBox(d.i, { x1, y1, x2: x1 + w, y2: y1 + h });
    } else {
      setBox(d.i, { x1: o.x1, y1: o.y1, x2: Math.min(W, Math.max(o.x1 + MIN, o.x2 + dx)), y2: Math.min(H, Math.max(o.y1 + MIN, o.y2 + dy)) });
    }
  };
  const end = () => {
    drag.current = null;
  };
  return { svgRef, start, move, end };
}

function DraggableBox({ b, color, label, onMove, onResize, dashed }: { b: Box; color: string; label: string; onMove: (e: RPointerEvent) => void; onResize: (e: RPointerEvent) => void; dashed?: boolean }) {
  return (
    <g>
      <rect x={b.x1} y={b.y1} width={b.x2 - b.x1} height={b.y2 - b.y1} fill="transparent" stroke={color} strokeWidth={2.2} strokeDasharray={dashed ? "6 3" : undefined} style={{ cursor: "move" }} onPointerDown={onMove} />
      <rect x={b.x2 - 7} y={b.y2 - 7} width={11} height={11} fill={color} stroke="var(--surface)" strokeWidth={1} style={{ cursor: "nwse-resize" }} onPointerDown={onResize} />
      <text x={b.x1 + 3} y={b.y1 - 4} fontSize={10} fill={color} fontFamily="var(--font-mono, monospace)" style={{ pointerEvents: "none" }}>
        {label}
      </text>
    </g>
  );
}

/** Keyboard/pointer-free fallback: sliders for the selected box. */
function BoxSliders({ boxes, names, setBox }: { boxes: Box[]; names: string[]; setBox: (i: number, b: Box) => void }) {
  const [sel, setSel] = useState(0);
  const i = Math.min(sel, boxes.length - 1);
  const b = boxes[i];
  const w = b.x2 - b.x1;
  const h = b.y2 - b.y1;
  return (
    <details className="text-xs">
      <summary className="cursor-pointer text-muted">Move boxes with sliders</summary>
      <div className="mt-2 space-y-1.5">
        <Seg label="Box" value={i} options={names.map((n, k) => ({ v: k, label: n }))} onChange={setSel} />
        <Slider label="x1" value={b.x1} min={0} max={W - w} step={1} onChange={(v) => setBox(i, { x1: v, y1: b.y1, x2: v + w, y2: b.y2 })} fmt={(v) => String(Math.round(v))} />
        <Slider label="y1" value={b.y1} min={0} max={H - h} step={1} onChange={(v) => setBox(i, { x1: b.x1, y1: v, x2: b.x2, y2: v + h })} fmt={(v) => String(Math.round(v))} />
        <Slider label="width" value={w} min={MIN} max={W - b.x1} step={1} onChange={(v) => setBox(i, { ...b, x2: b.x1 + v })} fmt={(v) => String(Math.round(v))} />
        <Slider label="height" value={h} min={MIN} max={H - b.y1} step={1} onChange={(v) => setBox(i, { ...b, y2: b.y1 + v })} fmt={(v) => String(Math.round(v))} />
      </div>
    </details>
  );
}

function Scene({ svgRef, label, onMove, onUp, children }: { svgRef: RefObject<SVGSVGElement | null>; label: string; onMove: (e: RPointerEvent) => void; onUp: () => void; children: ReactNode }) {
  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${W} ${H}`}
      className="h-auto w-full select-none rounded-md border border-border"
      style={{ maxWidth: 560, touchAction: "none", background: "var(--surface-2)" }}
      role="img"
      aria-label={label}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerLeave={onUp}
      onPointerCancel={onUp}
    >
      {children}
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/* (a) IoU                                                              */
/* ------------------------------------------------------------------ */

const IOU_STEPS = ["The two boxes", "Area of each box", "Intersection", "Union", "IoU"];

function IouMode({ gts, setGt }: { gts: Box[]; setGt: (i: number, b: Box) => void }) {
  const [pred, setPred] = useState<Box>({ x1: 70, y1: 90, x2: 176, y2: 200 });
  const boxes = [gts[0], pred];
  const setBox = (i: number, b: Box) => (i === 0 ? setGt(0, b) : setPred(b));
  const { svgRef, start, move, end } = useDrag(boxes, setBox);
  const s = useStepper(IOU_STEPS.length, { baseMs: 1300 });
  const p = iouParts(gts[0], pred);
  const st = s.step;
  const quality = p.iou >= 0.75 ? "a strict match (COCO's AP75 threshold)" : p.iou >= 0.5 ? "a match at the usual PASCAL VOC / mAP@0.5 threshold" : p.iou > 0 ? "below 0.5, so it counts as a false positive at mAP@0.5" : "no overlap at all. IoU is 0 no matter how far apart the boxes are";
  const narr = [
    <>Ground truth (the correct box) <b>G</b> {fmtBox(gts[0])} and prediction <b>P</b> {fmtBox(pred)}, written as corners (x1, y1, x2, y2). Drag either box, or its corner handle.</>,
    <>|G| = {r1(gts[0].x2 - gts[0].x1)} × {r1(gts[0].y2 - gts[0].y1)} = {r1(p.areaA)}; |P| = {r1(pred.x2 - pred.x1)} × {r1(pred.y2 - pred.y1)} = {r1(p.areaB)}.</>,
    p.interBox ? (
      <>The overlap (intersection) box: take the max of the top-left corners and the min of the bottom-right corners → {fmtBox(p.interBox)}, area {r1(p.inter)}.</>
    ) : (
      <>max(x1) ≥ min(x2) or max(y1) ≥ min(y2), so the boxes do not overlap. Intersection = 0.</>
    ),
    <>Union = |G| + |P| − |G∩P| = {r1(p.areaA)} + {r1(p.areaB)} − {r1(p.inter)} = {r1(p.union)}. We subtract the overlap so it is not counted twice.</>,
    <>IoU = {r1(p.inter)} / {r1(p.union)} = <b>{fmt(p.iou)}</b>: {quality}.</>,
  ];
  return (
    <SimShell
      title="Intersection over Union"
      subtitle="How much a predicted box overlaps the true box, divided by the area they cover together."
      controls={<Controls s={s} />}
      aside={
        <div className="space-y-3">
          <div>
            <div className="mb-1 font-medium">Live values</div>
            <Stat label="|G|" value={r1(p.areaA)} />
            <Stat label="|P|" value={r1(p.areaB)} />
            <Stat label="|G ∩ P|" value={r1(p.inter)} />
            <Stat label="|G ∪ P|" value={r1(p.union)} />
            <Stat label="IoU" value={<span className="text-accent">{fmt(p.iou)}</span>} />
            <Stat label="match @0.5 / @0.75" value={`${p.iou >= 0.5 ? "yes" : "no"} / ${p.iou >= 0.75 ? "yes" : "no"}`} />
          </div>
          <Narration title={`Step ${st + 1}: ${IOU_STEPS[st]}`}>{narr[st]}</Narration>
        </div>
      }
    >
      <Scene svgRef={svgRef} label="Ground-truth and predicted boxes with their intersection shaded" onMove={move} onUp={end}>
        {st >= 3 && (
          <>
            <rect x={gts[0].x1} y={gts[0].y1} width={gts[0].x2 - gts[0].x1} height={gts[0].y2 - gts[0].y1} fill="color-mix(in srgb, var(--accent) 14%, transparent)" />
            <rect x={pred.x1} y={pred.y1} width={pred.x2 - pred.x1} height={pred.y2 - pred.y1} fill="color-mix(in srgb, var(--accent) 14%, transparent)" />
          </>
        )}
        {st === 1 && (
          <>
            <rect x={gts[0].x1} y={gts[0].y1} width={gts[0].x2 - gts[0].x1} height={gts[0].y2 - gts[0].y1} fill="color-mix(in srgb, var(--accent) 12%, transparent)" />
            <rect x={pred.x1} y={pred.y1} width={pred.x2 - pred.x1} height={pred.y2 - pred.y1} fill="color-mix(in srgb, var(--clay) 12%, transparent)" />
          </>
        )}
        {st >= 2 && p.interBox && <rect x={p.interBox.x1} y={p.interBox.y1} width={p.interBox.x2 - p.interBox.x1} height={p.interBox.y2 - p.interBox.y1} fill="color-mix(in srgb, var(--gold) 55%, transparent)" stroke="var(--gold)" strokeWidth={1} />}
        <DraggableBox b={gts[0]} color="var(--accent)" label="G (ground truth)" onMove={start(0, "move")} onResize={start(0, "resize")} />
        <DraggableBox b={pred} color="var(--clay)" label="P (prediction)" onMove={start(1, "move")} onResize={start(1, "resize")} dashed />
        <text x={W - 6} y={H - 8} textAnchor="end" fontSize={13} fill="var(--text)" fontFamily="var(--font-mono, monospace)">
          IoU = {fmt(p.iou)}
        </text>
      </Scene>
      <div className="mt-2">
        <BoxSliders boxes={boxes} names={["G", "P"]} setBox={setBox} />
      </div>
    </SimShell>
  );
}

/* ------------------------------------------------------------------ */
/* (b) Anchors                                                          */
/* ------------------------------------------------------------------ */

function AnchorsMode({ gts, setGt }: { gts: Box[]; setGt: (i: number, b: Box) => void }) {
  const [stride, setStride] = useState(32);
  const [scales, setScales] = useState<number[]>([48, 96]);
  const [ratios, setRatios] = useState<number[]>([0.5, 1, 2]);
  const [posThr, setPosThr] = useState(0.5);
  const [negThr, setNegThr] = useState(0.4);
  const [force, setForce] = useState(true);
  const [showNeg, setShowNeg] = useState(false);
  const { svgRef, start, move, end } = useDrag(gts, setGt);
  const toggle = (arr: number[], set: (v: number[]) => void) => (v: number) => {
    if (arr.includes(v)) {
      if (arr.length > 1) set(arr.filter((x) => x !== v));
    } else set([...arr, v].sort((a, b) => a - b));
  };
  const anchors = useMemo(() => makeAnchors(W, H, stride, scales, ratios), [stride, scales, ratios]);
  const neg = Math.min(negThr, posThr);
  const matches = useMemo(() => matchAnchors(anchors, gts, posThr, neg, force), [anchors, gts, posThr, neg, force]);
  const counts = { pos: 0, neg: 0, ignore: 0 };
  matches.forEach((m) => counts[m.label]++);
  // Stepper walks through positives (best IoU first), then ignored anchors.
  const visit = useMemo(() => {
    const idx = matches.map((_, i) => i);
    const pos = idx.filter((i) => matches[i].label === "pos").sort((a, b) => matches[b].bestIou - matches[a].bestIou);
    const ign = idx.filter((i) => matches[i].label === "ignore").sort((a, b) => matches[b].bestIou - matches[a].bestIou);
    return [...pos, ...ign];
  }, [matches]);
  const s = useStepper(Math.max(1, visit.length), { baseMs: 1100 });
  const cur = visit.length ? visit[Math.min(s.step, visit.length - 1)] : -1;
  const a = cur >= 0 ? anchors[cur] : null;
  const m = cur >= 0 ? matches[cur] : null;
  const t = a && m && m.gt >= 0 ? encode(gts[m.gt], a.box) : null;
  const ac = a ? toCxcywh(a.box) : null;
  const gc = m && m.gt >= 0 ? toCxcywh(gts[m.gt]) : null;
  const rows = Math.floor(H / stride);
  const cols = Math.floor(W / stride);
  const labelColor = (l: string) => (l === "pos" ? "var(--accent)" : l === "ignore" ? "var(--gold)" : "var(--text-muted)");

  return (
    <SimShell
      title="Anchors and target assignment"
      subtitle="Every grid cell holds anchors (preset boxes) of several sizes and shapes. Each gets a label from its best IoU with a true box."
      controls={
        <div className="space-y-2">
          <Controls s={s} />
          <div className="flex flex-wrap gap-x-4 gap-y-2">
            <Seg label="Stride" value={stride} options={[32, 64].map((v) => ({ v, label: `${v} px` }))} onChange={(v) => { setStride(v); s.reset(); }} />
            <Toggle label="Scales √area" values={scales} options={[32, 48, 96, 160]} onToggle={(v) => { toggle(scales, setScales)(v); s.reset(); }} />
            <Toggle label="Ratios h/w" values={ratios} options={[0.5, 1, 2]} onToggle={(v) => { toggle(ratios, setRatios)(v); s.reset(); }} />
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            <Slider label="Positive IoU ≥" value={posThr} min={0.3} max={0.9} step={0.05} onChange={(v) => { setPosThr(v); s.reset(); }} />
            <Slider label="Negative IoU <" value={negThr} min={0.05} max={0.9} step={0.05} onChange={(v) => { setNegThr(v); s.reset(); }} />
          </div>
          <div className="flex flex-wrap gap-3 text-xs">
            <label className="flex items-center gap-1.5">
              <input type="checkbox" checked={force} onChange={(e) => { setForce(e.target.checked); s.reset(); }} /> each true box keeps its best anchor
            </label>
            <label className="flex items-center gap-1.5">
              <input type="checkbox" checked={showNeg} onChange={(e) => setShowNeg(e.target.checked)} /> draw negatives
            </label>
          </div>
        </div>
      }
      aside={
        <div className="space-y-3">
          <div>
            <div className="mb-1 font-medium">Assignment</div>
            <Stat label="grid" value={`${cols} × ${rows} cells`} />
            <Stat label="anchors / cell" value={`${scales.length} × ${ratios.length} = ${scales.length * ratios.length}`} />
            <Stat label="total anchors" value={anchors.length} />
            <Stat label="positive" value={<span className="text-accent">{counts.pos}</span>} />
            <Stat label="ignored" value={counts.ignore} />
            <Stat label="negative" value={counts.neg} />
            {negThr > posThr && <div className="mt-1 text-xs text-clay">Negative threshold capped at the positive one.</div>}
          </div>
          {a && m && (
            <div>
              <div className="mb-1 font-medium">Selected anchor</div>
              <Stat label="cell (row, col)" value={`(${a.row}, ${a.col})`} />
              <Stat label="scale, ratio" value={`${a.scale}, ${a.ratio}`} />
              <Stat label="w × h" value={`${r1(a.box.x2 - a.box.x1)} × ${r1(a.box.y2 - a.box.y1)}`} />
              <Stat label="best IoU" value={`${fmt(m.bestIou)} (GT ${m.gt + 1})`} />
              <Stat label="label" value={<span style={{ color: labelColor(m.label) }}>{m.label}{m.forced ? " (forced)" : ""}</span>} />
              {t && m.label === "pos" && (
                <>
                  <Stat label="tx" value={fmt(t.tx)} />
                  <Stat label="ty" value={fmt(t.ty)} />
                  <Stat label="tw" value={fmt(t.tw)} />
                  <Stat label="th" value={fmt(t.th)} />
                </>
              )}
            </div>
          )}
          <Narration title={a && m ? `Anchor ${s.step + 1} of ${visit.length}` : "No positive or ignored anchors"}>
            {a && m && ac && gc ? (
              m.label === "pos" ? (
                <>
                  {m.forced ? <>Its IoU {fmt(m.bestIou)} is below {posThr.toFixed(2)}. But it is GT {m.gt + 1}&apos;s best anchor, so we make it positive. </> : <>IoU {fmt(m.bestIou)} ≥ {posThr.toFixed(2)} → positive for GT {m.gt + 1}. </>}
                  Targets: tx = ({r1(gc.cx)} − {r1(ac.cx)}) / {r1(ac.w)} = {fmt(t!.tx)}, ty = ({r1(gc.cy)} − {r1(ac.cy)}) / {r1(ac.h)} = {fmt(t!.ty)}, tw = ln({r1(gc.w)} / {r1(ac.w)}) = {fmt(t!.tw)}, th = ln({r1(gc.h)} / {r1(ac.h)}) = {fmt(t!.th)}. The network learns to output these four numbers for this anchor: how far to shift and stretch it.
                </>
              ) : (
                <>IoU {fmt(m.bestIou)} is between {neg.toFixed(2)} and {posThr.toFixed(2)}. That is unclear, so this anchor is <b>ignored</b>. It adds nothing to either the class loss or the box loss.</>
              )
            ) : (
              <>Lower the positive threshold, add scales closer to the object sizes, or drag a box.</>
            )}
          </Narration>
        </div>
      }
    >
      <Scene svgRef={svgRef} label="Anchor grid over the image with positive, ignored and negative anchors" onMove={move} onUp={end}>
        {Array.from({ length: cols + 1 }, (_, c) => (
          <line key={`v${c}`} x1={c * stride} y1={0} x2={c * stride} y2={rows * stride} stroke="var(--border)" strokeWidth={0.6} />
        ))}
        {Array.from({ length: rows + 1 }, (_, r) => (
          <line key={`h${r}`} x1={0} y1={r * stride} x2={cols * stride} y2={r * stride} stroke="var(--border)" strokeWidth={0.6} />
        ))}
        {anchors.map((an, i) => {
          const mm = matches[i];
          if (mm.label === "neg" && !showNeg) return null;
          if (i === cur) return null;
          return (
            <rect key={i} x={an.box.x1} y={an.box.y1} width={an.box.x2 - an.box.x1} height={an.box.y2 - an.box.y1} fill="none" stroke={labelColor(mm.label)} strokeWidth={mm.label === "neg" ? 0.4 : 1} strokeOpacity={mm.label === "neg" ? 0.35 : 0.8} strokeDasharray={mm.label === "ignore" ? "3 2" : undefined} style={{ pointerEvents: "none" }} />
          );
        })}
        {Array.from({ length: rows * cols }, (_, k) => (
          <circle key={`c${k}`} cx={((k % cols) + 0.5) * stride} cy={(Math.floor(k / cols) + 0.5) * stride} r={1.6} fill="var(--text-muted)" style={{ pointerEvents: "none" }} />
        ))}
        {gts.map((g, i) => (
          <DraggableBox key={i} b={g} color="var(--text)" label={`GT ${i + 1}`} onMove={start(i, "move")} onResize={start(i, "resize")} />
        ))}
        {a && <rect x={a.box.x1} y={a.box.y1} width={a.box.x2 - a.box.x1} height={a.box.y2 - a.box.y1} fill="color-mix(in srgb, var(--gold) 18%, transparent)" stroke="var(--gold)" strokeWidth={2.4} style={{ pointerEvents: "none" }} />}
        {ac && gc && m?.label === "pos" && <line x1={ac.cx} y1={ac.cy} x2={gc.cx} y2={gc.cy} stroke="var(--gold)" strokeWidth={1.5} style={{ pointerEvents: "none" }} />}
      </Scene>
      <div className="mt-1 flex flex-wrap gap-3 text-xs text-muted">
        <span><span className="text-accent">━</span> positive</span>
        <span style={{ color: "var(--gold)" }}>┅ ignored</span>
        <span>━ negative (optional)</span>
        <span style={{ color: "var(--gold)" }}>■ selected anchor → its GT centre</span>
      </div>
      <div className="mt-2">
        <BoxSliders boxes={gts} names={gts.map((_, i) => `GT ${i + 1}`)} setBox={setGt} />
      </div>
    </SimShell>
  );
}

/* ------------------------------------------------------------------ */
/* (c) NMS                                                              */
/* ------------------------------------------------------------------ */

function NmsMode({ gts, setGt }: { gts: Box[]; setGt: (i: number, b: Box) => void }) {
  const [thr, setThr] = useState(0.5);
  const [scoreThr, setScoreThr] = useState(0.2);
  const [seed, setSeed] = useState(3);
  const { svgRef, start, move, end } = useDrag(gts, setGt);
  const cands: Candidate[] = useMemo(() => makeCandidates(gts, seed, 5, 3, W, H), [gts, seed]);
  const live = useMemo(() => cands.map((_, i) => i).filter((i) => cands[i].score >= scoreThr), [cands, scoreThr]);
  const trace = useMemo(() => nmsTrace(cands, live, thr), [cands, live, thr]);
  const total = trace.events.length + 1;
  const s = useStepper(total, { baseMs: 1200 });
  const st = Math.min(s.step, total - 1);

  // replay the trace up to the current step
  const kept = new Set<number>();
  const suppressed = new Map<number, number>(); // idx -> IoU with the box that suppressed it
  let current = -1;
  let lastComps: { idx: number; iou: number; suppressed: boolean }[] = [];
  for (let k = 0; k < st; k++) {
    const ev = trace.events[k];
    if (ev.kind === "pick") {
      kept.add(ev.idx);
      current = ev.idx;
      lastComps = [];
    } else {
      ev.comps.forEach((c) => c.suppressed && suppressed.set(c.idx, c.iou));
      lastComps = ev.comps;
    }
  }
  const done = st === total - 1;
  const ev = st > 0 ? trace.events[st - 1] : null;
  const before = precisionRecall(cands, live, gts, 0.5);
  const after = precisionRecall(cands, trace.keep, gts, 0.5);
  const compMap = new Map(lastComps.map((c) => [c.idx, c]));

  const color = (i: number) => {
    if (kept.has(i)) return i === current && !done ? "var(--gold)" : "var(--accent)";
    if (suppressed.has(i)) return "var(--clay)";
    return "var(--text-muted)";
  };

  return (
    <SimShell
      title="Greedy non-maximum suppression"
      subtitle="Keep the highest-scoring box. Delete every other box that overlaps it more than the threshold. Repeat."
      controls={
        <div className="space-y-2">
          <Controls s={s} />
          <div className="grid gap-2 sm:grid-cols-2">
            <Slider label="NMS IoU thr" value={thr} min={0.1} max={0.9} step={0.05} onChange={(v) => { setThr(v); s.reset(); }} />
            <Slider label="Score thr" value={scoreThr} min={0} max={0.8} step={0.05} onChange={(v) => { setScoreThr(v); s.reset(); }} />
          </div>
          <button className="sim-btn" onClick={() => { setSeed((v) => v + 1); s.reset(); }}>
            New candidates
          </button>
        </div>
      }
      aside={
        <div className="space-y-3">
          <div>
            <div className="mb-1 font-medium">Candidates by score</div>
            <div className="max-h-48 overflow-y-auto pr-1 font-mono text-xs">
              {trace.order.map((i, rank) => {
                const c = compMap.get(i);
                return (
                  <div key={i} className="flex justify-between gap-2 border-b border-border py-0.5" style={{ color: color(i) }}>
                    <span>#{rank + 1} s={cands[i].score.toFixed(2)}</span>
                    <span>{kept.has(i) ? "kept" : suppressed.has(i) ? `supp. ${fmt(suppressed.get(i)!, 2)}` : c ? `IoU ${fmt(c.iou, 2)}` : "—"}</span>
                  </div>
                );
              })}
            </div>
          </div>
          <div>
            <div className="mb-1 font-medium">Precision / recall @ IoU 0.5</div>
            <Stat label="before NMS" value={`P ${fmt(before.precision, 2)}  R ${fmt(before.recall, 2)}`} />
            <Stat label="  TP / FP / FN" value={`${before.tp} / ${before.fp} / ${before.fn}`} />
            <Stat label="after NMS" value={`P ${fmt(after.precision, 2)}  R ${fmt(after.recall, 2)}`} />
            <Stat label="  TP / FP / FN" value={`${after.tp} / ${after.fp} / ${after.fn}`} />
          </div>
          <Narration title={done ? "Finished" : ev ? (ev.kind === "pick" ? "Pick the highest remaining score" : "Compare and remove") : "Start"}>
            {!ev && <>{live.length} candidate boxes have score ≥ {scoreThr.toFixed(2)}. Sort them by score, then press Step.</>}
            {ev && ev.kind === "pick" && !done && <>Box #{trace.order.indexOf(ev.idx) + 1} (score {cands[ev.idx].score.toFixed(2)}) is the best of the {ev.remaining.length} boxes left → we <b>keep</b> it.</>}
            {ev && ev.kind === "suppress" && !done && (
              <>
                IoU with each remaining box: {ev.comps.length ? ev.comps.map((c) => `${fmt(c.iou, 2)}${c.suppressed ? "✕" : ""}`).join(", ") : "none left"}. Those above {thr.toFixed(2)} (✕) are suppressed (removed): {ev.comps.filter((c) => c.suppressed).length} removed, {ev.comps.filter((c) => !c.suppressed).length} remain.
              </>
            )}
            {done && (
              <>
                Kept {trace.keep.length} of {live.length} boxes. Precision rose from {fmt(before.precision, 2)} to {fmt(after.precision, 2)}, because duplicate boxes count as false positives. Recall is {fmt(after.recall, 2)}. Try a threshold of 0.1 (nearby objects may wrongly merge) or 0.9 (duplicates survive).
              </>
            )}
          </Narration>
        </div>
      }
    >
      <Scene svgRef={svgRef} label="Candidate detections, kept boxes and suppressed boxes" onMove={move} onUp={end}>
        {gts.map((g, i) => (
          <DraggableBox key={i} b={g} color="var(--text)" label={`GT ${i + 1}`} onMove={start(i, "move")} onResize={start(i, "resize")} dashed />
        ))}
        {trace.order.map((i) => {
          const b = cands[i].box;
          const isSup = suppressed.has(i);
          const isKept = kept.has(i);
          const inComp = compMap.has(i);
          return (
            <g key={i} style={{ pointerEvents: "none" }}>
              <rect x={b.x1} y={b.y1} width={b.x2 - b.x1} height={b.y2 - b.y1} fill={i === current && !done ? "color-mix(in srgb, var(--gold) 15%, transparent)" : "none"} stroke={color(i)} strokeWidth={isKept ? 2.4 : inComp ? 1.6 : 1} strokeOpacity={isSup ? 0.45 : 1} strokeDasharray={isSup ? "2 3" : undefined} />
              {(isKept || inComp) && (
                <text x={b.x2 - 2} y={b.y1 + 10} textAnchor="end" fontSize={9} fill={color(i)} fontFamily="var(--font-mono, monospace)">
                  {isKept ? cands[i].score.toFixed(2) : fmt(compMap.get(i)!.iou, 2)}
                </text>
              )}
            </g>
          );
        })}
      </Scene>
      <div className="mt-1 flex flex-wrap gap-3 text-xs text-muted">
        <span style={{ color: "var(--gold)" }}>■ current pick</span>
        <span className="text-accent">━ kept</span>
        <span className="text-clay">┅ suppressed</span>
        <span>━ not yet decided</span>
      </div>
      <div className="mt-2">
        <BoxSliders boxes={gts} names={gts.map((_, i) => `GT ${i + 1}`)} setBox={setGt} />
      </div>
    </SimShell>
  );
}

/* ------------------------------------------------------------------ */
/* Root                                                                 */
/* ------------------------------------------------------------------ */

type Mode = "iou" | "anchors" | "nms";

export default function DetectionSim() {
  const [mode, setMode] = useState<Mode>("iou");
  const [gts, setGts] = useState<Box[]>(DEFAULT_GT);
  const [nGt, setNGt] = useState(3);
  const shown = gts.slice(0, nGt);
  const setGt = (i: number, b: Box) => setGts((prev) => prev.map((g, k) => (k === i ? b : g)));
  const tabs: { id: Mode; label: string }[] = [
    { id: "iou", label: "IoU" },
    { id: "anchors", label: "Anchors" },
    { id: "nms", label: "NMS" },
  ];
  return (
    <div className="min-w-0">
      <div className="mb-[-0.75rem] mt-6 flex flex-wrap items-center gap-2" role="tablist" aria-label="Detection concept">
        {tabs.map((t) => (
          <button key={t.id} role="tab" aria-selected={mode === t.id} className={`sim-btn ${mode === t.id ? "primary" : ""}`} onClick={() => setMode(t.id)}>
            {t.label}
          </button>
        ))}
        {mode !== "iou" && (
          <span className="ml-auto flex items-center gap-1.5 text-xs text-muted">
            objects
            {[2, 3].map((n) => (
              <button key={n} className={`sim-btn ${nGt === n ? "primary" : ""}`} aria-pressed={nGt === n} onClick={() => setNGt(n)}>
                {n}
              </button>
            ))}
            <button className="sim-btn" onClick={() => setGts(DEFAULT_GT)}>
              Reset boxes
            </button>
          </span>
        )}
      </div>
      {mode === "iou" && <IouMode gts={gts} setGt={setGt} />}
      {mode === "anchors" && <AnchorsMode gts={shown} setGt={setGt} />}
      {mode === "nms" && <NmsMode gts={shown} setGt={setGt} />}
    </div>
  );
}
