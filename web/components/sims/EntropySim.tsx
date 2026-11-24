"use client";

import { useEffect, useRef, useState, type PointerEvent as RPointerEvent } from "react";
import { Controls, Narration, SimShell, Slider, fmt, signedColor, useStepper } from "./SimShell";
import { softmax } from "@/lib/sim/random";
import {
  GAUSS_LIMITS,
  PRESETS,
  P_FLOOR,
  gaussQ,
  klTerm,
  logitsFromQ,
  measures,
  objectiveValue,
  preset,
  setBar,
  stepFree,
  stepGauss,
  toUnit,
  type Family,
  type Objective,
  type PresetName,
  type Unit,
} from "@/lib/sim/entropy";

const STEPS_PER_TICK = 5;
const HIST_MAX = 400;

interface Model {
  q: number[];
  z: number[]; // logits of the free family
  mu: number;
  s: number; // log sigma of the Gaussian family
  iter: number;
  hist: number[]; // objective value (nats) after each step
  q0: number[]; // snapshot restored by Reset
  mu0: number;
  s0: number;
}

function freshModel(q: number[], mu: number, s: number, family: Family): Model {
  const qq = family === "gauss" ? gaussQ(q.length, mu, s) : q;
  return { q: qq, z: logitsFromQ(qq), mu, s, iter: 0, hist: [], q0: qq, mu0: mu, s0: s };
}

const defaultMu = (K: number) => Math.max(0, (K - 1) * 0.28);

/* ------------------------------------------------------------------ bar chart */

const BW = 300;
const BH = 170;
const BM = { l: 30, r: 8, t: 14, b: 24 };

function Bars({
  dist,
  other,
  color,
  otherColor,
  name,
  onSet,
}: {
  dist: number[];
  other: number[];
  color: string;
  otherColor: string;
  name: string;
  onSet: ((i: number, v: number) => void) | null;
}) {
  const K = dist.length;
  const pw = BW - BM.l - BM.r;
  const ph = BH - BM.t - BM.b;
  const slot = pw / K;
  const sy = (v: number) => BM.t + (1 - v) * ph;
  const dragIdx = useRef<number | null>(null);

  const locate = (e: RPointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * BW;
    const y = ((e.clientY - r.top) / r.height) * BH;
    const i = Math.max(0, Math.min(K - 1, Math.floor((x - BM.l) / slot)));
    const v = Math.max(0, Math.min(1, 1 - (y - BM.t) / ph));
    return { i, v };
  };

  return (
    <svg
      viewBox={`0 0 ${BW} ${BH}`}
      className={`h-auto w-full select-none ${onSet ? "cursor-ns-resize touch-none" : ""}`}
      role="img"
      aria-label={`Bar chart of distribution ${name}${onSet ? ", drag a bar to change it" : ""}`}
      onPointerDown={(e) => {
        if (!onSet) return;
        const { i, v } = locate(e);
        dragIdx.current = i;
        e.currentTarget.setPointerCapture(e.pointerId);
        onSet(i, v);
      }}
      onPointerMove={(e) => {
        if (!onSet || dragIdx.current === null) return;
        onSet(dragIdx.current, locate(e).v);
      }}
      onPointerUp={() => (dragIdx.current = null)}
      onPointerCancel={() => (dragIdx.current = null)}
    >
      <rect x={BM.l} y={BM.t} width={pw} height={ph} fill="none" stroke="var(--border)" />
      {[0, 0.5, 1].map((t) => (
        <g key={t}>
          <line x1={BM.l} x2={BM.l + pw} y1={sy(t)} y2={sy(t)} stroke="var(--border)" strokeDasharray={t === 0.5 ? "2 3" : undefined} />
          <text x={BM.l - 4} y={sy(t) + 3} fontSize={9} textAnchor="end" fill="var(--text-muted)">
            {t}
          </text>
        </g>
      ))}
      {dist.map((v, i) => {
        const x = BM.l + i * slot + slot * 0.16;
        const w = slot * 0.68;
        return (
          <g key={i}>
            <rect x={x} y={sy(v)} width={w} height={Math.max(0, sy(0) - sy(v))} fill={color} opacity={0.85} />
            <line x1={x - 2} x2={x + w + 2} y1={sy(other[i])} y2={sy(other[i])} stroke={otherColor} strokeWidth={2} strokeDasharray="4 2" />
            <text x={x + w / 2} y={Math.max(BM.t + 8, sy(v) - 3)} fontSize={8.5} textAnchor="middle" fill="var(--text)">
              {v.toFixed(2)}
            </text>
            <text x={x + w / 2} y={BH - 9} fontSize={10} textAnchor="middle" fill="var(--text-muted)">
              {i + 1}
            </text>
          </g>
        );
      })}
      <text x={BM.l + 4} y={BM.t + 10} fontSize={11} fill="var(--text-faint)">
        {name}
      </text>
    </svg>
  );
}

/* ------------------------------------------------------- stacked decomposition */

function Decomposition({ Hp, klPQ, Hpq, unit, K }: { Hp: number; klPQ: number; Hpq: number; unit: Unit; K: number }) {
  const W = 560;
  const L = 8;
  const pw = W - 2 * L;
  const infinite = !Number.isFinite(klPQ);
  const cap = Math.max(toUnit(Math.log(K), unit) * 1.5, infinite ? 0 : toUnit(Hpq, unit) * 1.05, 0.1);
  const sx = (v: number) => L + (Math.min(v, cap) / cap) * pw;
  const hp = toUnit(Hp, unit);
  const hpq = infinite ? cap : toUnit(Hpq, unit);
  const u = unit === "bits" ? "bits" : "nats";
  return (
    <svg viewBox={`0 0 ${W} 64`} className="h-auto w-full" role="img" aria-label={`Cross-entropy ${fmt(hpq, 3)} ${u} equals entropy ${fmt(hp, 3)} plus KL divergence`}>
      <rect x={sx(0)} y={8} width={Math.max(0, sx(hp) - sx(0))} height={20} fill="var(--accent)" opacity={0.85} />
      <rect x={sx(hp)} y={8} width={Math.max(0, sx(hpq) - sx(hp))} height={20} fill="var(--clay)" opacity={0.85} />
      <rect x={L} y={8} width={pw} height={20} fill="none" stroke="var(--border)" />
      <line x1={sx(toUnit(Math.log(K), unit))} x2={sx(toUnit(Math.log(K), unit))} y1={4} y2={32} stroke="var(--gold)" strokeDasharray="3 2" />
      <text x={sx(toUnit(Math.log(K), unit))} y={44} fontSize={9.5} textAnchor="middle" fill="var(--gold)">
        log K
      </text>
      <text x={L} y={58} fontSize={10.5} fill="var(--accent)">
        H(p) = {fmt(hp, 3)}
      </text>
      <text x={W / 2} y={58} fontSize={10.5} textAnchor="middle" fill="var(--clay)">
        + KL(p‖q) = {infinite ? "∞" : fmt(toUnit(klPQ, unit), 3)}
      </text>
      <text x={W - L} y={58} fontSize={10.5} textAnchor="end" fill="var(--text)">
        = H(p,q) = {infinite ? "∞" : fmt(hpq, 3)} {u}
      </text>
    </svg>
  );
}

function Spark({ hist, unit }: { hist: number[]; unit: Unit }) {
  const W = 260, H = 56;
  if (hist.length < 2) return <div className="text-xs text-faint">Press Play or Step to train q.</div>;
  const vals = hist.map((v) => toUnit(v, unit));
  const max = Math.max(...vals, 1e-9);
  const pts = vals.map((v, i) => `${((i / (vals.length - 1)) * (W - 4) + 2).toFixed(1)},${(H - 4 - (v / max) * (H - 10)).toFixed(1)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Objective during training">
      <rect x={0.5} y={0.5} width={W - 1} height={H - 1} fill="none" stroke="var(--border)" />
      <polyline points={pts} fill="none" stroke="var(--clay)" strokeWidth={1.8} />
      <text x={4} y={11} fontSize={9} fill="var(--text-faint)">
        max {fmt(max, 3)}
      </text>
    </svg>
  );
}

/* ----------------------------------------------------------------- simulator */

export default function EntropySim() {
  const [K, setK] = useState(8);
  const [unit, setUnit] = useState<Unit>("bits");
  const [family, setFamily] = useState<Family>("free");
  const [objective, setObjective] = useState<Objective>("forward");
  const [lr, setLr] = useState(1);
  const [pName, setPName] = useState<PresetName | "custom">("bimodal");
  const [qName, setQName] = useState<PresetName | "custom">("uniform");
  const [p, setP] = useState<number[]>(() => preset("bimodal", 8));
  const [m, setM] = useState<Model>(() => freshModel(preset("uniform", 8), defaultMu(8), 0, "free"));
  const [editing, setEditing] = useState<"p" | "q">("p");

  const trainTick = () => {
    setM((prev) => {
      let { z, mu, s } = prev;
      const hist = prev.hist.slice();
      let q = prev.q;
      for (let k = 0; k < STEPS_PER_TICK; k++) {
        if (family === "free") {
          z = stepFree(z, p, objective, lr);
          q = softmax(z);
        } else {
          const r = stepGauss(mu, s, p, objective, lr);
          mu = r.mu;
          s = r.s;
          q = gaussQ(p.length, mu, s);
        }
        hist.push(objectiveValue(p, q, objective));
      }
      return { ...prev, z, mu, s, q, iter: prev.iter + STEPS_PER_TICK, hist: hist.slice(-HIST_MAX) };
    });
  };

  const st = useStepper(Infinity, { baseMs: 160, onTick: trainTick });

  // Reset (step back to 0) restores q to its state before training started.
  useEffect(() => {
    if (st.step !== 0) return;
    setM((prev) => (prev.iter === 0 ? prev : { ...prev, q: prev.q0, z: logitsFromQ(prev.q0), mu: prev.mu0, s: prev.s0, iter: 0, hist: [] }));
  }, [st.step]);

  const restart = (next: Model) => {
    st.reset();
    setM(next);
  };

  const changeK = (k: number) => {
    setK(k);
    const pn = pName === "custom" ? "bimodal" : pName;
    const qn = qName === "custom" ? "uniform" : qName;
    setPName(pn);
    setQName(qn);
    setP(preset(pn, k));
    restart(freshModel(preset(qn, k), defaultMu(k), 0, family));
  };

  const editP = (i: number, v: number) => {
    setPName("custom");
    setP((prev) => setBar(prev, i, v));
    st.setPlaying(false);
    setM((prev) => ({ ...prev, iter: 0, hist: [], q0: prev.q, mu0: prev.mu, s0: prev.s }));
  };

  const editQ = (i: number, v: number) => {
    setQName("custom");
    st.setPlaying(false);
    setM((prev) => {
      const q = setBar(prev.q, i, v);
      return { ...prev, q, z: logitsFromQ(q), iter: 0, hist: [], q0: q };
    });
  };

  const setGauss = (mu: number, s: number) => {
    st.setPlaying(false);
    restart(freshModel(m.q, mu, s, "gauss"));
  };

  const changeFamily = (f: Family) => {
    setFamily(f);
    setLr(f === "free" ? 1 : 0.5);
    if (f === "gauss") {
      setQName("custom");
      restart(freshModel(m.q, defaultMu(K), 0, "gauss"));
    } else restart(freshModel(m.q, m.mu, m.s, "free"));
  };

  const ms = measures(p, m.q);
  const u = (v: number) => (Number.isFinite(v) ? fmt(toUnit(v, unit), 4) : "∞");
  const uName = unit === "bits" ? "bits" : "nats";
  const obj = objectiveValue(p, m.q, objective);
  const pHasZero = p.some((x) => x === 0);
  const sigma = Math.exp(m.s);

  /* ---------- narration ---------- */
  let story;
  if (m.iter === 0) {
    story = (
      <>
        <p>
          H(p) = {u(ms.Hp)} {uName} is the average surprise of outcomes drawn from p when you code them with p itself; the maximum possible for K = {K} is log K = {fmt(toUnit(Math.log(K), unit), 4)}.
          Coding them with q instead costs H(p,q) = {u(ms.Hpq)} {uName}; the excess, KL(p‖q) = {u(ms.klPQ)}, is the price of believing q when the truth is p.
        </p>
        <p className="mt-1">
          {!Number.isFinite(ms.klPQ)
            ? "KL(p‖q) is infinite: some outcome has p > 0 but q = 0, so its surprise −log q is unbounded."
            : `Note the asymmetry: KL(q‖p) = ${u(ms.klQP)}. JS = ${u(ms.js)} is symmetric and never exceeds ${unit === "bits" ? "1 bit" : "log 2 ≈ 0.693 nats"}.`}{" "}
          Drag bars to edit, then press Play to train q.
        </p>
      </>
    );
  } else {
    const what = objective === "forward" ? "KL(p‖q) (the same as minimising H(p,q), since H(p) is fixed)" : "KL(q‖p)";
    const gaussNote =
      family === "gauss"
        ? objective === "forward"
          ? ` q is a single bump (μ = ${fmt(m.mu + 1, 2)} on the 1…K axis, σ = ${fmt(sigma, 2)}). Forward KL punishes q for missing any mass of p, so the bump moves between the modes and widens to cover them: mode-covering.`
          : ` q is a single bump (μ = ${fmt(m.mu + 1, 2)}, σ = ${fmt(sigma, 2)}). Reverse KL punishes q for putting mass where p is small, so the bump tends to settle on one mode and stay narrow: mode-seeking. Which mode depends on where μ starts.`
        : obj < 1e-3
          ? " q has (almost) become p: with K free logits the model family contains p, so both directions of KL reach 0."
          : " With K free logits the family contains p (if p has full support), so the minimum is q = p.";
    story = (
      <>
        <p>
          Iteration {m.iter}: gradient descent (η = {lr.toFixed(2)}) on q&apos;s logits, minimising {what}. Current value {u(obj)} {uName}.
        </p>
        <p className="mt-1">
          {gaussNote}
          {objective === "reverse" && pHasZero ? ` p has zero-probability outcomes; log p is floored at log(${P_FLOOR}) for the gradient, which pushes q's mass there towards 0.` : ""}
        </p>
      </>
    );
  }

  const rows = p.map((pi, i) => {
    const qi = m.q[i];
    return { i, pi, qi, surpr: qi > 0 ? -Math.log(qi) : Infinity, term: klTerm(pi, qi) };
  });
  const maxTerm = Math.max(1e-9, ...rows.map((r) => (Number.isFinite(r.term) ? Math.abs(r.term) : 0)));

  return (
    <SimShell
      title="Entropy, Cross-Entropy & KL Divergence"
      subtitle="p is the true distribution, q the model. Drag bars, compare the measures, then train q by gradient descent."
      controls={
        <div className="space-y-2.5">
          <Controls s={st} canPrev={false} extra={<span className="ml-auto font-mono text-xs text-faint">iter {m.iter}</span>} />
          <div className="flex flex-wrap items-center gap-1.5 text-xs">
            <span className="text-muted">units</span>
            {(["bits", "nats"] as const).map((v) => (
              <button key={v} className={`sim-btn ${unit === v ? "primary" : ""}`} onClick={() => setUnit(v)} aria-pressed={unit === v}>
                {v}
              </button>
            ))}
            <span className="ml-2 text-muted">model q</span>
            <button className={`sim-btn ${family === "free" ? "primary" : ""}`} onClick={() => changeFamily("free")} aria-pressed={family === "free"}>
              free (K logits)
            </button>
            <button className={`sim-btn ${family === "gauss" ? "primary" : ""}`} onClick={() => changeFamily("gauss")} aria-pressed={family === "gauss"}>
              unimodal (μ, σ)
            </button>
            <span className="ml-2 text-muted">minimise</span>
            <button className={`sim-btn ${objective === "forward" ? "primary" : ""}`} onClick={() => { setObjective("forward"); restart({ ...m, iter: 0, hist: [], q0: m.q, mu0: m.mu, s0: m.s }); }} aria-pressed={objective === "forward"}>
              KL(p‖q) forward
            </button>
            <button className={`sim-btn ${objective === "reverse" ? "primary" : ""}`} onClick={() => { setObjective("reverse"); restart({ ...m, iter: 0, hist: [], q0: m.q, mu0: m.mu, s0: m.s }); }} aria-pressed={objective === "reverse"}>
              KL(q‖p) reverse
            </button>
          </div>
          <div className="grid gap-1.5 sm:grid-cols-2">
            <Slider label="outcomes K" value={K} min={2} max={8} step={1} onChange={(v) => changeK(v)} fmt={(v) => String(v)} />
            <Slider label="learning rate η" value={lr} min={0.05} max={3} step={0.05} onChange={setLr} />
            {family === "gauss" && (
              <>
                <Slider label="q centre μ" value={m.mu + 1} min={0} max={K + 1} step={0.05} onChange={(v) => setGauss(v - 1, m.s)} />
                <Slider label="q width σ" value={sigma} min={GAUSS_LIMITS.minSigma} max={K} step={0.05} onChange={(v) => setGauss(m.mu, Math.log(v))} />
              </>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-1.5 text-xs">
            <span className="w-14 text-muted">p preset</span>
            {PRESETS.map((n) => (
              <button key={n} className={`sim-btn ${pName === n ? "primary" : ""}`} onClick={() => { setPName(n); setP(preset(n, K)); restart({ ...m, iter: 0, hist: [], q0: m.q, mu0: m.mu, s0: m.s }); }} aria-pressed={pName === n}>
                {n}
              </button>
            ))}
          </div>
          {family === "free" && (
            <div className="flex flex-wrap items-center gap-1.5 text-xs">
              <span className="w-14 text-muted">q preset</span>
              {PRESETS.map((n) => (
                <button key={n} className={`sim-btn ${qName === n ? "primary" : ""}`} onClick={() => { setQName(n); restart(freshModel(preset(n, K), m.mu, m.s, "free")); }} aria-pressed={qName === n}>
                  {n}
                </button>
              ))}
            </div>
          )}
        </div>
      }
      aside={
        <div className="space-y-3">
          <Narration title={m.iter === 0 ? "Comparing p and q" : family === "gauss" ? "Training a restricted q" : "Training q"}>{story}</Narration>
          <table className="w-full text-xs">
            <tbody className="font-mono">
              {[
                ["H(p)", ms.Hp, "entropy of the truth"],
                ["H(q)", ms.Hq, "entropy of the model"],
                ["H(p,q)", ms.Hpq, "cross-entropy"],
                ["KL(p‖q)", ms.klPQ, "forward KL"],
                ["KL(q‖p)", ms.klQP, "reverse KL"],
                ["JS(p,q)", ms.js, "Jensen–Shannon"],
              ].map(([label, v, d]) => (
                <tr key={label as string} className="border-t border-border">
                  <td className="py-1 pr-2">{label as string}</td>
                  <td className="py-1 pr-2 text-right">{u(v as number)}</td>
                  <td className="py-1 font-sans text-faint">{d as string}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div>
            <div className="mb-1 text-xs text-muted">
              objective {objective === "forward" ? "KL(p‖q)" : "KL(q‖p)"} ({uName})
            </div>
            <Spark hist={m.hist} unit={unit} />
          </div>
        </div>
      }
    >
      <div className="grid gap-3 sm:grid-cols-2">
        <div>
          <Bars dist={p} other={m.q} color="var(--accent)" otherColor="var(--clay)" name="p (true) — dashes show q" onSet={editP} />
        </div>
        <div>
          <Bars dist={m.q} other={p} color="var(--clay)" otherColor="var(--accent)" name={family === "gauss" ? "q (unimodal model) — dashes show p" : "q (model) — dashes show p"} onSet={family === "free" ? editQ : null} />
        </div>
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-1.5 text-xs">
        <span className="text-muted">fine-edit</span>
        <button className={`sim-btn ${editing === "p" ? "primary" : ""}`} onClick={() => setEditing("p")} aria-pressed={editing === "p"}>
          p
        </button>
        <button className={`sim-btn ${editing === "q" ? "primary" : ""}`} onClick={() => setEditing("q")} aria-pressed={editing === "q"} disabled={family === "gauss"}>
          q
        </button>
        <span className="text-faint">Changing one outcome rescales the others so the total stays 1.</span>
      </div>
      <div className="mt-1.5 grid gap-1 sm:grid-cols-2">
        {(editing === "p" || family === "gauss" ? p : m.q).map((v, i) => (
          <Slider key={i} label={`${editing === "p" || family === "gauss" ? "p" : "q"}${i + 1}`} value={v} min={0} max={1} step={0.01} onChange={(nv) => (editing === "p" || family === "gauss" ? editP(i, nv) : editQ(i, nv))} />
        ))}
      </div>

      <div className="mt-4">
        <div className="mb-1 text-xs text-muted">Cross-entropy = entropy + KL divergence</div>
        <Decomposition Hp={ms.Hp} klPQ={ms.klPQ} Hpq={ms.Hpq} unit={unit} K={K} />
      </div>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-muted">
              <th className="py-1 pr-2 font-medium">i</th>
              <th className="py-1 pr-2 text-right font-medium">p_i</th>
              <th className="py-1 pr-2 text-right font-medium">q_i</th>
              <th className="py-1 pr-2 text-right font-medium">−log q_i</th>
              <th className="py-1 text-right font-medium">p_i·log(p_i/q_i)</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {rows.map((r) => (
              <tr key={r.i} className="border-t border-border">
                <td className="py-1 pr-2">{r.i + 1}</td>
                <td className="py-1 pr-2 text-right">{fmt(r.pi, 4)}</td>
                <td className="py-1 pr-2 text-right">{fmt(r.qi, 4)}</td>
                <td className="py-1 pr-2 text-right">{Number.isFinite(r.surpr) ? fmt(toUnit(r.surpr, unit), 3) : "∞"}</td>
                <td className="py-1 text-right" style={{ background: Number.isFinite(r.term) ? signedColor(r.term, maxTerm) : "color-mix(in srgb, var(--clay) 60%, transparent)" }}>
                  {Number.isFinite(r.term) ? fmt(toUnit(r.term, unit), 4) : "∞"}
                </td>
              </tr>
            ))}
            <tr className="border-t border-border font-semibold">
              <td className="py-1 pr-2 font-sans" colSpan={4}>
                sum = KL(p‖q) ({uName})
              </td>
              <td className="py-1 text-right">{u(ms.klPQ)}</td>
            </tr>
          </tbody>
        </table>
        <p className="mt-1 text-xs text-faint">
          Individual terms can be negative (where q_i &gt; p_i) but their sum never is. Entries with p_i = 0 contribute exactly 0.
        </p>
      </div>
    </SimShell>
  );
}
