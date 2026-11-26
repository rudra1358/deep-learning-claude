"use client";

import { useId, useState } from "react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import { bce, focal, hinge, huber, logCosh, logit, mae, mse, mseProb, sigmoid, type LossValue } from "@/lib/sim/losses";

const N = 120; // marker positions 0..N
const VW = 560;
const M = { l: 40, r: 12, t: 12, b: 26 };
const PW = VW - M.l - M.r;

interface Series {
  key: string;
  label: string;
  color: string;
  dash?: string;
  fn: (x: number) => LossValue;
}

function Plot({
  series,
  enabled,
  xDomain,
  yDomain,
  height,
  marker,
  pick,
  xLabel,
  yLabel,
  ariaLabel,
}: {
  series: Series[];
  enabled: Record<string, boolean>;
  xDomain: [number, number];
  yDomain: [number, number];
  height: number;
  marker: number;
  pick: "loss" | "grad";
  xLabel: string;
  yLabel: string;
  ariaLabel: string;
}) {
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const PH = height - M.t - M.b;
  const [xa, xb] = xDomain;
  const [ya, yb] = yDomain;
  const sx = (x: number) => M.l + ((x - xa) / (xb - xa)) * PW;
  const sy = (y: number) => M.t + ((yb - y) / (yb - ya)) * PH;
  const cy = (y: number) => sy(Math.max(ya - (yb - ya), Math.min(yb + (yb - ya), y)));
  const xs = Array.from({ length: 241 }, (_, i) => xa + ((xb - xa) * i) / 240);
  const xTicks = Array.from({ length: 5 }, (_, i) => xa + ((xb - xa) * i) / 4);
  const yTicks = [ya, (ya + yb) / 2, yb];
  return (
    <svg viewBox={`0 0 ${VW} ${height}`} className="h-auto w-full" role="img" aria-label={ariaLabel}>
      <defs>
        <clipPath id={`lc-${uid}`}>
          <rect x={M.l} y={M.t} width={PW} height={PH} />
        </clipPath>
      </defs>
      <rect x={M.l} y={M.t} width={PW} height={PH} fill="none" stroke="var(--border)" />
      {ya < 0 && yb > 0 && <line x1={M.l} x2={M.l + PW} y1={sy(0)} y2={sy(0)} stroke="var(--border)" />}
      {xa < 0 && xb > 0 && <line x1={sx(0)} x2={sx(0)} y1={M.t} y2={M.t + PH} stroke="var(--border)" />}
      {xTicks.map((t) => (
        <text key={t} x={sx(t)} y={height - 9} textAnchor="middle" fontSize={10} fill="var(--text-muted)">
          {+t.toFixed(2)}
        </text>
      ))}
      {yTicks.map((t) => (
        <text key={t} x={M.l - 5} y={sy(t) + 3} textAnchor="end" fontSize={10} fill="var(--text-muted)">
          {+t.toFixed(2)}
        </text>
      ))}
      <text x={M.l + PW - 4} y={M.t + PH - 5} textAnchor="end" fontSize={11} fill="var(--text-faint)">
        {xLabel}
      </text>
      <text x={M.l + 5} y={M.t + 12} fontSize={11} fill="var(--text-faint)">
        {yLabel}
      </text>
      <g clipPath={`url(#lc-${uid})`}>
        {series
          .filter((s) => enabled[s.key])
          .map((s) => (
            <polyline
              key={s.key}
              points={xs.map((x) => `${sx(x).toFixed(1)},${cy(s.fn(x)[pick]).toFixed(1)}`).join(" ")}
              fill="none"
              stroke={s.color}
              strokeWidth={2}
              strokeDasharray={s.dash}
              strokeLinejoin="round"
            />
          ))}
        <line x1={sx(marker)} x2={sx(marker)} y1={M.t} y2={M.t + PH} stroke="var(--gold)" strokeDasharray="3 3" />
        {series
          .filter((s) => enabled[s.key])
          .map((s) => (
            <circle key={s.key} cx={sx(marker)} cy={cy(s.fn(marker)[pick])} r={4} fill={s.color} stroke="var(--surface)" strokeWidth={1.2} />
          ))}
      </g>
    </svg>
  );
}

function Legend({ series, enabled, setEnabled }: { series: Series[]; enabled: Record<string, boolean>; setEnabled: (e: Record<string, boolean>) => void }) {
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
      {series.map((s) => (
        <label key={s.key} className="flex items-center gap-1.5">
          <input type="checkbox" checked={enabled[s.key]} onChange={(e) => setEnabled({ ...enabled, [s.key]: e.target.checked })} aria-label={`Show ${s.label}`} />
          <svg width="22" height="8" aria-hidden="true">
            <line x1="1" y1="4" x2="21" y2="4" stroke={s.color} strokeWidth={2.2} strokeDasharray={s.dash} />
          </svg>
          {s.label}
        </label>
      ))}
    </div>
  );
}

function Readout({ series, enabled, at, gradLabel }: { series: Series[]; enabled: Record<string, boolean>; at: number; gradLabel: string }) {
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-left text-muted">
          <th className="py-1 pr-2 font-medium">Loss</th>
          <th className="py-1 pr-2 font-medium">value</th>
          <th className="py-1 font-medium">{gradLabel}</th>
        </tr>
      </thead>
      <tbody className="font-mono">
        {series.map((s) => {
          const v = s.fn(at);
          return (
            <tr key={s.key} className={`border-t border-border ${enabled[s.key] ? "" : "opacity-40"}`}>
              <td className="py-1 pr-2 font-sans">
                <span className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle" style={{ background: s.color }} />
                {s.label}
              </td>
              <td className="py-1 pr-2">{fmt(v.loss, 4)}</td>
              <td className="py-1">{fmt(v.grad, 4)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

const COLORS = ["var(--accent)", "var(--clay)", "var(--gold)", "var(--text-muted)"];

export default function LossSim() {
  const [tab, setTab] = useState<"reg" | "cls">("reg");
  const [delta, setDelta] = useState(1);
  const [gamma, setGamma] = useState(2);
  const [y, setY] = useState<0 | 1>(1);
  const [axis, setAxis] = useState<"logit" | "prob">("logit");
  const [showGrad, setShowGrad] = useState(true);
  const [regOn, setRegOn] = useState<Record<string, boolean>>({ mse: true, mae: true, huber: true, logcosh: true });
  const [clsOn, setClsOn] = useState<Record<string, boolean>>({ bce: true, focal: true, hinge: true, mseprob: true });

  const s = useStepper(N + 1, { baseMs: 120 });
  const frac = s.step / N;

  /* ---------- regression: x = residual r ---------- */
  const regSeries: Series[] = [
    { key: "mse", label: "MSE  r²", color: COLORS[0], fn: mse },
    { key: "mae", label: "MAE  |r|", color: COLORS[1], fn: mae },
    { key: "huber", label: `Huber (δ=${delta.toFixed(2)})`, color: COLORS[2], fn: (r) => huber(r, delta) },
    { key: "logcosh", label: "Log-cosh", color: COLORS[3], dash: "6 3", fn: logCosh },
  ];
  const rAt = -3 + 6 * frac;

  /* ---------- classification: x = logit z, or probability p (converted to z) ---------- */
  const toZ = (x: number) => (axis === "logit" ? x : logit(Math.min(1 - 1e-6, Math.max(1e-6, x))));
  const wrap = (f: (z: number) => LossValue) => (x: number) => f(toZ(x));
  const clsSeries: Series[] = [
    { key: "bce", label: "Binary CE", color: COLORS[0], fn: wrap((z) => bce(z, y)) },
    { key: "focal", label: `Focal (γ=${gamma.toFixed(1)})`, color: COLORS[2], fn: wrap((z) => focal(z, y, gamma)) },
    { key: "hinge", label: "Hinge (margin)", color: COLORS[1], fn: wrap((z) => hinge(z, y)) },
    { key: "mseprob", label: "MSE on σ(z)", color: COLORS[3], dash: "6 3", fn: wrap((z) => mseProb(z, y)) },
  ];
  const xDomainCls: [number, number] = axis === "logit" ? [-6, 6] : [0.005, 0.995];
  const xAtCls = xDomainCls[0] + (xDomainCls[1] - xDomainCls[0]) * frac;
  const zAt = toZ(xAtCls);
  const pAt = sigmoid(zAt);

  const isReg = tab === "reg";
  const series = isReg ? regSeries : clsSeries;
  const enabled = isReg ? regOn : clsOn;
  const at = isReg ? rAt : xAtCls;

  let story;
  if (isReg) {
    const hv = huber(rAt, delta);
    story = (
      <>
        <p>
          Residual (how far off the prediction is) r = ŷ − y = <strong className="text-text">{fmt(rAt, 2)}</strong>.
        </p>
        <p className="mt-1">
          MSE&apos;s gradient 2r = {fmt(2 * rAt, 2)} grows with the error, so one outlier (a far-off point) can drown out the rest of a batch. MAE&apos;s gradient is always ±1. Huber acts like MSE for |r| ≤ δ and like MAE beyond that:
          {Math.abs(rAt) <= delta ? ` here |r| ≤ δ, so its gradient is r = ${fmt(hv.grad, 2)}.` : ` here |r| > δ, so its gradient stops growing at ±δ = ${fmt(hv.grad, 2)}.`} Log-cosh&apos;s gradient tanh(r) = {fmt(Math.tanh(rAt), 3)} is a smooth version of that limit.
        </p>
      </>
    );
  } else {
    const pt = y === 1 ? pAt : 1 - pAt;
    const b = bce(zAt, y);
    const m = mseProb(zAt, y);
    story = (
      <>
        <p>
          Logit z = {fmt(zAt, 2)} → p = σ(z) = {fmt(pAt, 3)}. The true label is y = {y}, so the probability given to the right answer is p_t = <strong className="text-text">{fmt(pt, 3)}</strong>
          {pt < 0.5 ? " (wrong side)." : " (correct side)."}
        </p>
        <p className="mt-1">
          BCE gradient p − y = {fmt(b.grad, 3)}. MSE-on-probability gradient = {fmt(m.grad, 3)}. Its extra factor p(1 − p) = {fmt(pAt * (1 - pAt), 3)} makes it
          {pt < 0.1 ? " almost vanish when the model is confidently wrong." : " small wherever the sigmoid curve goes flat."} Focal loss multiplies BCE by (1 − p_t)^γ = {fmt((1 - pt) ** gamma, 3)}
          {pt < 0.5 ? ": a hard example keeps most of its loss." : ": an easy example counts for less."}
          {hinge(zAt, y).loss === 0 ? " Hinge is exactly zero: the margin (how far past the boundary) is over 1." : ""}
        </p>
      </>
    );
  }

  return (
    <SimShell
      title="Loss Function Explorer"
      subtitle="See how each loss and its gradient change as the prediction moves. Play slides the prediction marker."
      controls={
        <div className="space-y-2.5">
          <Controls s={s} canPrev={false} />
          <div className="flex flex-wrap items-center gap-1.5">
            <button className={`sim-btn ${isReg ? "primary" : ""}`} onClick={() => setTab("reg")} aria-pressed={isReg}>
              Regression
            </button>
            <button className={`sim-btn ${!isReg ? "primary" : ""}`} onClick={() => setTab("cls")} aria-pressed={!isReg}>
              Classification
            </button>
            <label className="ml-2 flex items-center gap-1.5 text-xs">
              <input type="checkbox" checked={showGrad} onChange={(e) => setShowGrad(e.target.checked)} aria-label="Show derivative" />
              show slope
            </label>
            {!isReg && (
              <>
                <span className="ml-2 text-xs text-muted">true label</span>
                {([0, 1] as const).map((v) => (
                  <button key={v} className={`sim-btn ${y === v ? "primary" : ""}`} onClick={() => setY(v)} aria-pressed={y === v}>
                    y = {v}
                  </button>
                ))}
                <span className="ml-2 text-xs text-muted">x-axis</span>
                <button className={`sim-btn ${axis === "logit" ? "primary" : ""}`} onClick={() => setAxis("logit")} aria-pressed={axis === "logit"}>
                  logit z
                </button>
                <button className={`sim-btn ${axis === "prob" ? "primary" : ""}`} onClick={() => setAxis("prob")} aria-pressed={axis === "prob"}>
                  probability p
                </button>
              </>
            )}
          </div>
          <div className="grid gap-1.5 sm:grid-cols-2">
            <Slider
              label={isReg ? "residual r" : axis === "logit" ? "logit z" : "probability p"}
              value={s.step}
              min={0}
              max={N}
              step={1}
              onChange={(v) => s.setStep(v)}
              fmt={() => fmt(at, 2)}
            />
            {isReg ? (
              <Slider label="Huber δ" value={delta} min={0.1} max={3} step={0.05} onChange={setDelta} />
            ) : (
              <Slider label="focal γ" value={gamma} min={0} max={5} step={0.1} onChange={setGamma} fmt={(v) => v.toFixed(1)} />
            )}
          </div>
          <Legend series={series} enabled={enabled} setEnabled={isReg ? setRegOn : setClsOn} />
        </div>
      }
      aside={
        <div className="space-y-3">
          <Narration title={isReg ? "Regression losses" : "Classification losses"}>{story}</Narration>
          <Readout series={series} enabled={enabled} at={at} gradLabel={isReg ? "dL/dr" : "dL/dz"} />
          {!isReg && <p className="text-xs text-faint">Gradients are taken with respect to the logit z (the raw score), because that is what backpropagation sends into the network. Focal loss uses α = 1 here.</p>}
        </div>
      }
    >
      <Plot
        series={series}
        enabled={enabled}
        xDomain={isReg ? [-3, 3] : xDomainCls}
        yDomain={isReg ? [0, 4] : [0, 5]}
        height={240}
        marker={at}
        pick="loss"
        xLabel={isReg ? "residual r = ŷ − y" : axis === "logit" ? "logit z" : "p = σ(z)"}
        yLabel="loss"
        ariaLabel="Loss as a function of the prediction"
      />
      {showGrad && (
        <Plot
          series={series}
          enabled={enabled}
          xDomain={isReg ? [-3, 3] : xDomainCls}
          yDomain={isReg ? [-3, 3] : [-1.2, 1.2]}
          height={170}
          marker={at}
          pick="grad"
          xLabel={isReg ? "residual r" : axis === "logit" ? "logit z" : "p = σ(z)"}
          yLabel={isReg ? "dL/dr" : "dL/dz"}
          ariaLabel="Derivative of each loss"
        />
      )}
    </SimShell>
  );
}
