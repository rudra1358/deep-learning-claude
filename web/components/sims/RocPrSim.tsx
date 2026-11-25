"use client";

import { useMemo, useState, type PointerEvent as RPointerEvent, type ReactNode } from "react";
import { Shuffle } from "lucide-react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import { bestF1, confusionAt, curves, histogram, makeScores, metricsFrom, type CurvePoint } from "@/lib/sim/roc-pr";

const N = 1000; // scores drawn per configuration
const T_MAX = 4; // threshold sweep: +4 → −4 in steps of 0.1
const T_STEP = 0.1;
const STEPS = Math.round((2 * T_MAX) / T_STEP) + 1; // 81
const tAt = (k: number) => Math.round((T_MAX - k * T_STEP) * 10) / 10;
const kAt = (t: number) => Math.max(0, Math.min(STEPS - 1, Math.round((T_MAX - t) / T_STEP)));

/* histogram geometry */
const HW = 320, HH = 140, HP = { l: 26, r: 8, t: 10, b: 20 };
const LO = -5, HI = 5, BINS = 40;
const hx = (v: number) => HP.l + ((v - LO) / (HI - LO)) * (HW - HP.l - HP.r);
const hxInv = (px: number) => LO + ((px - HP.l) / (HW - HP.l - HP.r)) * (HI - LO);

/* curve panels */
const CW = 160, CH = 170, CP = { l: 24, r: 6, t: 8, b: 30 };
const cx = (v: number) => CP.l + v * (CW - CP.l - CP.r);
const cy = (v: number) => CP.t + (1 - v) * (CH - CP.t - CP.b);
const poly = (pts: CurvePoint[]) => pts.map((p, i) => `${i ? "L" : "M"}${cx(p.x).toFixed(1)},${cy(p.y).toFixed(1)}`).join(" ");

function Stat({ label, value, tone }: { label: ReactNode; value: ReactNode; tone?: string }) {
  return (
    <div className="flex justify-between gap-3 border-b border-border py-0.5 last:border-b-0">
      <span className="text-muted">{label}</span>
      <span className={`font-mono ${tone ?? ""}`}>{value}</span>
    </div>
  );
}

function Axes({ xLabel, yLabel }: { xLabel: string; yLabel: string }) {
  return (
    <g>
      <rect x={cx(0)} y={cy(1)} width={cx(1) - cx(0)} height={cy(0) - cy(1)} fill="none" stroke="var(--border)" />
      {[0, 0.5, 1].map((v) => (
        <g key={v}>
          <line x1={cx(0)} x2={cx(1)} y1={cy(v)} y2={cy(v)} stroke="var(--border)" strokeWidth={0.5} />
          <text x={cx(0) - 3} y={cy(v) + 3} fontSize="8" textAnchor="end" fill="var(--text-muted)">
            {v}
          </text>
          <text x={cx(v)} y={cy(0) + 10} fontSize="8" textAnchor="middle" fill="var(--text-muted)">
            {v}
          </text>
        </g>
      ))}
      <text x={(cx(0) + cx(1)) / 2} y={CH - 6} fontSize="8.5" textAnchor="middle" fill="var(--text-muted)">
        {xLabel}
      </text>
      <text x={7} y={(cy(0) + cy(1)) / 2} fontSize="8.5" textAnchor="middle" fill="var(--text-muted)" transform={`rotate(-90 7 ${(cy(0) + cy(1)) / 2})`}>
        {yLabel}
      </text>
    </g>
  );
}

export default function RocPrSim() {
  const [sep, setSep] = useState(2);
  const [posPct, setPosPct] = useState(30);
  const [seed, setSeed] = useState(7);
  const s = useStepper(STEPS, { baseMs: 220 });
  const { setStep, setPlaying } = s;
  const t = tAt(s.step);

  const set = useMemo(() => makeScores(sep, posPct / 100, N, seed), [sep, posPct, seed]);
  const cur = useMemo(() => curves(set), [set]);
  const hist = useMemo(() => histogram(set, LO, HI, BINS), [set]);
  const best = useMemo(() => bestF1(set), [set]);
  const m = metricsFrom(confusionAt(set, t));
  const prevalence = set.nPos / N;

  const maxCount = Math.max(1, ...hist.pos, ...hist.neg);
  const binW = (HW - HP.l - HP.r) / BINS;
  const hy = (c: number) => HH - HP.b - (c / maxCount) * (HH - HP.t - HP.b);

  const dragTo = (e: RPointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * HW;
    setPlaying(false);
    setStep(kAt(hxInv(px)));
  };

  /* ---------- narration (built from the computed numbers) ---------- */
  const P = set.nPos, Nn = set.nNeg;
  let title: string;
  let text: string;
  if (m.predictedPos === 0) {
    title = `t = ${t.toFixed(1)} — nothing is predicted positive`;
    text = `The threshold is above every score, so TP = FP = 0. Recall = 0/${P} = 0 and FPR = 0/${Nn} = 0: the ROC curve starts at the corner (0, 0). Precision = 0/0 is undefined, so there is no PR point yet. Press Play to lower the threshold and sweep out both curves.`;
  } else if (m.fn === 0 && m.tn === 0) {
    title = `t = ${t.toFixed(1)} — everything is predicted positive`;
    text = `Every example is flagged: recall = 1 and FPR = 1, the ROC corner (1, 1). Precision has fallen to the prevalence, ${P}/${N} = ${fmt(prevalence, 3)} — the dashed baseline of the PR plot, which is what a classifier with no skill achieves at any recall.`;
  } else {
    title = `t = ${t.toFixed(1)} — ${m.predictedPos} of ${N} flagged positive`;
    text = `Recall (TPR) = TP/(TP+FN) = ${m.tp}/${P} = ${fmt(m.tpr, 3)}; FPR = FP/(FP+TN) = ${m.fp}/${Nn} = ${fmt(m.fpr, 3)}; precision = TP/(TP+FP) = ${m.tp}/${m.predictedPos} = ${fmt(m.precision ?? 0, 3)}; F1 = ${fmt(m.f1, 3)}. `;
    if (posPct <= 10) {
      text += `With only ${P} positives against ${Nn} negatives, even this FPR of ${fmt(m.fpr * 100, 1)}% means ${m.fp} false alarms next to ${m.tp} true hits. The ROC curve barely notices (AUC ${fmt(cur.auc, 3)}), but precision and AP (${fmt(cur.ap, 3)}) do — this is why PR curves are preferred for rare positives.`;
    } else if (sep < 0.25) {
      text += `The two distributions almost coincide, so every threshold catches positives and negatives at the same rate: the ROC curve hugs the diagonal (AUC ${fmt(cur.auc, 3)} ≈ 0.5) and precision stays near the prevalence ${fmt(prevalence, 2)}.`;
    } else if (Math.abs(t - best.t) < 0.05 + 1e-9) {
      text += `This is (to the 0.1 grid) the F1-maximising threshold ${fmt(best.t, 2)}, F1 = ${fmt(best.f1, 3)}. A different cost of false positives vs false negatives would move the best threshold.`;
    } else if (t > best.t) {
      text += `The threshold is strict: few false positives, but ${m.fn} positives are missed. Lowering t raises recall and FPR together — you move up and to the right along the ROC curve.`;
    } else {
      text += `The threshold is lenient: ${m.fp} negatives are flagged. Raising t trades recall for precision; the F1 optimum here is near t = ${fmt(best.t, 2)}.`;
    }
  }

  const ptRoc = { x: m.fpr, y: m.tpr };
  const ptPr = m.precision === null ? null : { x: m.tpr, y: m.precision };
  const prDraw: CurvePoint[] = cur.pr.length ? [{ x: 0, y: cur.pr[0].y, t: Infinity }, ...cur.pr] : [];

  return (
    <SimShell
      title="Thresholds, ROC and PR Curves"
      subtitle={`A classifier's scores for ${N} examples: positives ~ N(+d/2, 1), negatives ~ N(−d/2, 1). Predict positive when score ≥ t. Drag on the histogram to move t.`}
      controls={
        <Controls
          s={s}
          extra={
            <button className="sim-btn" onClick={() => setSeed((v) => v + 1)} aria-label="Draw a new random sample of scores">
              <Shuffle size={14} /> Resample
            </button>
          }
        />
      }
      aside={
        <div className="space-y-3">
          <Slider label="separation d" value={sep} min={0} max={4} step={0.1} onChange={setSep} fmt={(v) => v.toFixed(1)} />
          <Slider label="positives %" value={posPct} min={2} max={50} step={1} onChange={setPosPct} fmt={(v) => `${v}%`} />
          <Slider
            label="threshold t"
            value={-t}
            min={-T_MAX}
            max={T_MAX}
            step={T_STEP}
            onChange={(v) => {
              setPlaying(false);
              setStep(kAt(-v));
            }}
            fmt={() => t.toFixed(1)}
          />
          <p className="text-[11px] text-muted">The threshold slider runs strict → lenient (left → right), the same direction as Play.</p>

          <div className="grid grid-cols-[auto_1fr_1fr] gap-1 text-center font-mono text-[11px]" role="table" aria-label="Confusion matrix">
            <div />
            <div className="text-muted">pred +</div>
            <div className="text-muted">pred −</div>
            <div className="pr-1 text-left text-muted">actual +</div>
            <div className="rounded border border-border p-1.5" style={{ background: "color-mix(in srgb, var(--accent) 22%, transparent)" }}>
              TP {m.tp}
            </div>
            <div className="rounded border border-border p-1.5" style={{ background: "color-mix(in srgb, var(--clay) 14%, transparent)" }}>
              FN {m.fn}
            </div>
            <div className="pr-1 text-left text-muted">actual −</div>
            <div className="rounded border border-border p-1.5" style={{ background: "color-mix(in srgb, var(--clay) 22%, transparent)" }}>
              FP {m.fp}
            </div>
            <div className="rounded border border-border p-1.5" style={{ background: "color-mix(in srgb, var(--accent) 10%, transparent)" }}>
              TN {m.tn}
            </div>
          </div>

          <div className="rounded-md bg-surface-2 p-2 text-[11px]">
            <Stat label="recall = TPR" value={fmt(m.tpr, 3)} tone="text-accent" />
            <Stat label="FPR" value={fmt(m.fpr, 3)} tone="text-clay" />
            <Stat label="precision" value={m.precision === null ? "undefined (0/0)" : fmt(m.precision, 3)} />
            <Stat label="F1" value={fmt(m.f1, 3)} />
            <Stat label="accuracy" value={fmt(m.accuracy, 3)} />
            <Stat label="ROC AUC" value={fmt(cur.auc, 3)} />
            <Stat label="average precision" value={fmt(cur.ap, 3)} />
            <Stat label="prevalence (P / n)" value={`${P}/${N} = ${fmt(prevalence, 3)}`} />
            <Stat label="best F1 threshold" value={`${fmt(best.t, 2)} (F1 ${fmt(best.f1, 3)})`} tone="text-gold" />
          </div>
          <Narration title={title}>{text}</Narration>
        </div>
      }
    >
      <figure>
        <svg
          viewBox={`0 0 ${HW} ${HH}`}
          className="h-auto w-full cursor-ew-resize touch-none select-none"
          role="img"
          aria-label={`Score histograms of ${P} positives and ${Nn} negatives with the threshold at ${t.toFixed(1)}`}
          onPointerDown={(e) => {
            e.currentTarget.setPointerCapture(e.pointerId);
            dragTo(e);
          }}
          onPointerMove={(e) => {
            if (e.buttons & 1) dragTo(e);
          }}
        >
          <rect x={hx(Math.max(LO, Math.min(HI, t)))} y={HP.t} width={Math.max(0, hx(HI) - hx(Math.max(LO, Math.min(HI, t))))} height={HH - HP.t - HP.b} fill="color-mix(in srgb, var(--gold) 9%, transparent)" />
          {hist.neg.map((c, i) => (
            <rect key={`n${i}`} x={HP.l + i * binW + 0.4} y={hy(c)} width={binW - 0.8} height={HH - HP.b - hy(c)} fill="color-mix(in srgb, var(--clay) 55%, transparent)" />
          ))}
          {hist.pos.map((c, i) => (
            <rect key={`p${i}`} x={HP.l + i * binW + 0.4} y={hy(c)} width={binW - 0.8} height={HH - HP.b - hy(c)} fill="color-mix(in srgb, var(--accent) 60%, transparent)" />
          ))}
          <line x1={HP.l} x2={HW - HP.r} y1={HH - HP.b} y2={HH - HP.b} stroke="var(--border)" />
          {[-4, -2, 0, 2, 4].map((v) => (
            <text key={v} x={hx(v)} y={HH - 7} fontSize="8.5" textAnchor="middle" fill="var(--text-muted)">
              {v}
            </text>
          ))}
          <text x={HP.l - 4} y={HP.t + 6} fontSize="8" textAnchor="end" fill="var(--text-muted)">
            {maxCount}
          </text>
          <text x={HP.l - 4} y={HH - HP.b} fontSize="8" textAnchor="end" fill="var(--text-muted)">
            0
          </text>
          <text x={HW - HP.r} y={HP.t + 6} fontSize="8.5" textAnchor="end" fill="var(--text-muted)">
            score →
          </text>
          {t >= LO && t <= HI && (
            <g>
              <line x1={hx(t)} x2={hx(t)} y1={HP.t - 4} y2={HH - HP.b} stroke="var(--gold)" strokeWidth={2} />
              <text x={hx(t) + (t > 3 ? -3 : 3)} y={HP.t + 4} fontSize="8.5" textAnchor={t > 3 ? "end" : "start"} fill="var(--gold)">
                t = {t.toFixed(1)}
              </text>
            </g>
          )}
        </svg>
        <figcaption className="mt-1 flex flex-wrap gap-3 text-xs text-muted">
          <span className="text-accent">■ positives ({P})</span>
          <span className="text-clay">■ negatives ({Nn})</span>
          <span className="text-gold">| threshold — shaded side is predicted positive</span>
        </figcaption>
      </figure>

      <div className="mt-3 grid grid-cols-2 gap-3">
        <figure className="min-w-0">
          <svg viewBox={`0 0 ${CW} ${CH}`} className="h-auto w-full" role="img" aria-label={`ROC curve, AUC ${fmt(cur.auc, 3)}, current point FPR ${fmt(m.fpr, 3)}, TPR ${fmt(m.tpr, 3)}`}>
            <Axes xLabel="FPR" yLabel="TPR (recall)" />
            <path d={`${poly(cur.roc)} L${cx(1)},${cy(0)} Z`} fill="color-mix(in srgb, var(--accent) 12%, transparent)" stroke="none" />
            <line x1={cx(0)} y1={cy(0)} x2={cx(1)} y2={cy(1)} stroke="var(--text-muted)" strokeDasharray="3 3" strokeWidth={0.8} />
            <path d={poly(cur.roc)} fill="none" stroke="var(--accent)" strokeWidth={1.6} />
            <circle cx={cx(ptRoc.x)} cy={cy(ptRoc.y)} r={4} fill="var(--gold)" stroke="var(--surface)" strokeWidth={1} />
            <text x={cx(0.97)} y={cy(0.06)} fontSize="8.5" textAnchor="end" fill="var(--text)">
              AUC {fmt(cur.auc, 3)}
            </text>
          </svg>
          <figcaption className="text-center text-[11px] text-muted">ROC · dashed = chance</figcaption>
        </figure>
        <figure className="min-w-0">
          <svg viewBox={`0 0 ${CW} ${CH}`} className="h-auto w-full" role="img" aria-label={`Precision-recall curve, average precision ${fmt(cur.ap, 3)}`}>
            <Axes xLabel="recall" yLabel="precision" />
            <line x1={cx(0)} x2={cx(1)} y1={cy(prevalence)} y2={cy(prevalence)} stroke="var(--text-muted)" strokeDasharray="3 3" strokeWidth={0.8} />
            <path d={poly(prDraw)} fill="none" stroke="var(--clay)" strokeWidth={1.4} />
            {ptPr && <circle cx={cx(ptPr.x)} cy={cy(ptPr.y)} r={4} fill="var(--gold)" stroke="var(--surface)" strokeWidth={1} />}
            <text x={cx(0.04)} y={cy(0.04)} fontSize="8.5" textAnchor="start" fill="var(--text)">
              AP {fmt(cur.ap, 3)}
            </text>
          </svg>
          <figcaption className="text-center text-[11px] text-muted">PR · dashed = prevalence</figcaption>
        </figure>
      </div>
    </SimShell>
  );
}
