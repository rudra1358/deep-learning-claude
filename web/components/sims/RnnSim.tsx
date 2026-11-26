"use client";

import { useEffect, useMemo, useState } from "react";
import { Controls, Narration, SimShell, Slider, fmt, signedColor, useStepper } from "./SimShell";
import { RNN_SUBSTEPS, makeRnnWeights, memoryProbe, parseChars, parseNumbers, runRnn, spectralNorm, type RnnStep, type RnnSubStep, type RnnWeights } from "@/lib/sim/rnn";

const SUB_LABEL: Record<RnnSubStep, string> = {
  input: "1 · Input term W_xh·x_t",
  recurrent: "2 · Recurrent term W_hh·h_{t−1}",
  sum: "3 · Sum z_t",
  tanh: "4 · tanh → h_t",
  output: "5 · Output head y_t",
};

const vec = (v: number[], d = 3) => "[" + v.map((x) => fmt(x, d)).join(", ") + "]";
const sci = (v: number) => (v === 0 ? "0" : v.toExponential(1));

/** Expand row `k` of A·v with every number substituted. */
function expandRow(A: number[][], v: number[], k: number, result: number) {
  return A[k].map((a, j) => `${fmt(a, 2)}×${fmt(v[j], 2)}`).join(" + ") + ` = ${fmt(result, 3)}`;
}

/** Signed horizontal bars, one per component, for the detail panel. */
function VecRow({ name, v, on, visible, maxAbs = 1.5 }: { name: string; v: number[]; on: boolean; visible: boolean; maxAbs?: number }) {
  return (
    <div className={`flex items-center gap-2 rounded px-1.5 py-0.5 ${on ? "bg-surface-2" : ""}`} style={on ? { outline: "1px solid var(--gold)" } : undefined}>
      <span className="w-20 shrink-0 font-mono text-[11px] text-muted">{name}</span>
      {visible ? (
        <span className="flex min-w-0 flex-wrap gap-1">
          {v.map((x, k) => (
            <span key={k} className="flex items-center gap-1">
              <svg viewBox="0 0 40 10" className="h-2.5 w-10" aria-hidden>
                <line x1={20} x2={20} y1={0} y2={10} stroke="var(--border)" />
                <rect x={x >= 0 ? 20 : 20 - Math.min(1, Math.abs(x) / maxAbs) * 20} y={1} width={Math.max(0.5, Math.min(1, Math.abs(x) / maxAbs) * 20)} height={8} fill={x >= 0 ? "var(--accent)" : "var(--clay)"} />
              </svg>
              <span className="rounded px-1 font-mono text-[11px]" style={{ background: signedColor(x, maxAbs) }}>
                {fmt(x, 3)}
              </span>
            </span>
          ))}
        </span>
      ) : (
        <span className="text-[11px] text-faint">not computed yet</span>
      )}
    </div>
  );
}

function Diagram({ steps, t, sub, labels, w }: { steps: RnnStep[]; t: number; sub: RnnSubStep; labels: string[]; w: RnnWeights }) {
  const T = steps.length;
  const colW = 92;
  const W = 70 + T * colW;
  const subIdx = RNN_SUBSTEPS.indexOf(sub);
  const hVisible = (tau: number) => tau < t || (tau === t && subIdx >= 3);
  const yVisible = (tau: number) => tau < t || (tau === t && subIdx >= 4);
  const barW = 9, gap = 3;
  return (
    <svg viewBox={`0 0 ${W} 232`} className="mx-auto h-auto w-full" style={{ minWidth: Math.min(W, 560), maxWidth: W * 1.25 }} role="img" aria-label={`Unrolled RNN over ${T} time steps, currently at step ${t + 1}`}>
      {/* h_0 */}
      <rect x={8} y={98} width={40} height={34} rx={6} fill="var(--surface-2)" stroke="var(--border)" />
      <text x={28} y={119} textAnchor="middle" fontSize={11} fill="var(--text-muted)">
        h₀=0
      </text>
      {steps.map((s, tau) => {
        const x0 = 64 + tau * colW;
        const cur = tau === t;
        const done = tau < t;
        const stroke = cur ? "var(--gold)" : done ? "var(--accent)" : "var(--border)";
        const fill = cur ? "color-mix(in srgb, var(--gold) 14%, var(--surface))" : "var(--surface)";
        const recOn = cur && (sub === "recurrent" || sub === "sum");
        const inOn = cur && (sub === "input" || sub === "sum");
        const outOn = cur && sub === "output";
        const prevX = tau === 0 ? 48 : x0 - colW + 64;
        return (
          <g key={tau}>
            {/* recurrent arrow h_{t-1} -> cell */}
            <line x1={prevX} x2={x0 - 3} y1={115} y2={115} stroke={recOn ? "var(--gold)" : done || cur ? "var(--accent)" : "var(--border)"} strokeWidth={recOn ? 2.6 : 1.4} />
            <polygon points={`${x0 - 3},115 ${x0 - 9},111 ${x0 - 9},119`} fill={recOn ? "var(--gold)" : done || cur ? "var(--accent)" : "var(--border)"} />
            {tau > 0 && (
              <text x={x0 - 14} y={108} textAnchor="end" fontSize={9} fill="var(--text-muted)">
                h{tau}
              </text>
            )}
            {/* input arrow */}
            <line x1={x0 + 32} x2={x0 + 32} y1={186} y2={152} stroke={inOn ? "var(--gold)" : "var(--text-muted)"} strokeWidth={inOn ? 2.6 : 1.2} />
            <polygon points={`${x0 + 32},151 ${x0 + 28},158 ${x0 + 36},158`} fill={inOn ? "var(--gold)" : "var(--text-muted)"} />
            <rect x={x0 + 8} y={188} width={48} height={24} rx={5} fill="var(--surface-2)" stroke={inOn ? "var(--gold)" : "var(--border)"} />
            <text x={x0 + 32} y={204} textAnchor="middle" fontSize={11} fill="var(--text)">
              {labels[tau]}
            </text>
            <text x={x0 + 32} y={226} textAnchor="middle" fontSize={9} fill={cur ? "var(--gold)" : "var(--text-muted)"}>
              t={tau + 1}
            </text>
            {/* cell */}
            <rect x={x0} y={80} width={64} height={70} rx={9} fill={fill} stroke={stroke} strokeWidth={cur ? 2.2 : 1.2} />
            <text x={x0 + 32} y={94} textAnchor="middle" fontSize={10} fill="var(--text-muted)">
              tanh
            </text>
            <line x1={x0 + 6} x2={x0 + 58} y1={124} y2={124} stroke="var(--border)" />
            {hVisible(tau) &&
              s.h.map((v, k) => {
                const n = s.h.length;
                const left = x0 + 32 - (n * (barW + gap) - gap) / 2 + k * (barW + gap);
                const hgt = Math.min(1, Math.abs(v)) * 22;
                return <rect key={k} x={left} y={v >= 0 ? 124 - hgt : 124} width={barW} height={Math.max(0.6, hgt)} fill={v >= 0 ? "var(--accent)" : "var(--clay)"} />;
              })}
            {/* output head */}
            <line x1={x0 + 32} x2={x0 + 32} y1={79} y2={46} stroke={outOn ? "var(--gold)" : yVisible(tau) ? "var(--text-muted)" : "var(--border)"} strokeWidth={outOn ? 2.6 : 1.2} />
            <polygon points={`${x0 + 32},44 ${x0 + 28},51 ${x0 + 36},51`} fill={outOn ? "var(--gold)" : yVisible(tau) ? "var(--text-muted)" : "var(--border)"} />
            <text x={x0 + 32} y={36} textAnchor="middle" fontSize={11} fill={yVisible(tau) ? "var(--text)" : "var(--text-muted)"}>
              {yVisible(tau) ? `y=${fmt(s.y, 2)}` : "y=?"}
            </text>
          </g>
        );
      })}
      <text x={8} y={16} fontSize={10} fill="var(--text-muted)">
        same W_xh, W_hh, b, W_hy in every cell · ‖W_hh‖₂ = {fmt(spectralNorm(w.Whh), 2)}
      </text>
    </svg>
  );
}

function ProbeChart({ dh, eps }: { dh: number[]; eps: number }) {
  const T = dh.length;
  const logs = dh.map((v) => (v > 0 ? Math.log10(v) : -16));
  const lo = Math.max(-16, Math.min(-4, Math.floor(Math.min(...logs))));
  const hi = Math.max(0, Math.ceil(Math.max(...logs, Math.log10(eps))));
  const W = 720;
  const colW = (W - 54) / T;
  const H = 150;
  const top = 18, bottom = 120;
  const yOf = (l: number) => bottom - ((l - lo) / (hi - lo || 1)) * (bottom - top);
  const ticks: number[] = [];
  for (let l = lo; l <= hi; l += Math.max(1, Math.round((hi - lo) / 4))) ticks.push(l);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" style={{ minWidth: 420 }} role="img" aria-label="Memory probe: size of the change in each hidden state after perturbing the first input, log scale">
      {ticks.map((l) => (
        <g key={l}>
          <line x1={44} x2={W - 4} y1={yOf(l)} y2={yOf(l)} stroke="var(--border)" strokeDasharray="2 3" />
          <text x={40} y={yOf(l) + 3} textAnchor="end" fontSize={9} fill="var(--text-muted)">
            1e{l}
          </text>
        </g>
      ))}
      <line x1={44} x2={W - 4} y1={yOf(Math.log10(eps))} y2={yOf(Math.log10(eps))} stroke="var(--gold)" strokeWidth={1.2} />
      <text x={W - 6} y={yOf(Math.log10(eps)) - 3} textAnchor="end" fontSize={9} fill="var(--gold)">
        ε = {fmt(eps, 2)}
      </text>
      {dh.map((v, tau) => {
        const x = 50 + tau * colW;
        const y = yOf(Math.max(lo, logs[tau]));
        return (
          <g key={tau}>
            <rect x={x + colW * 0.15} y={y} width={Math.min(colW * 0.7, 60)} height={Math.max(0.6, bottom - y)} fill="var(--accent)" opacity={0.85} />
            <text x={x + colW * 0.15 + Math.min(colW * 0.7, 60) / 2} y={Math.max(top - 4, y - 3)} textAnchor="middle" fontSize={8.5} fill="var(--text)">
              {sci(v)}
            </text>
            <text x={x + colW * 0.15 + Math.min(colW * 0.7, 60) / 2} y={bottom + 13} textAnchor="middle" fontSize={9} fill="var(--text-muted)">
              t={tau + 1}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export default function RnnSim() {
  const [mode, setMode] = useState<"numbers" | "chars">("numbers");
  const [numText, setNumText] = useState("1, 0, 0, 0, 0, 0, 0, 0");
  const [charText, setCharText] = useState("abcab");
  const [hidden, setHidden] = useState(3);
  const [rho, setRho] = useState(0.9);
  const [eps, setEps] = useState(0.5);

  const input = useMemo(() => {
    if (mode === "numbers") {
      const xs = parseNumbers(numText);
      const v = xs.length ? xs : [[0]];
      return { xs: v, labels: v.map((x) => fmt(x[0], 2)), vocab: [] as string[] };
    }
    const p = parseChars(charText);
    if (!p.xs.length) return { xs: [[1]], labels: ["?"], vocab: ["?"] };
    return { xs: p.xs, labels: p.chars.map((c) => `"${c}"`), vocab: p.vocab };
  }, [mode, numText, charText]);

  const I = input.xs[0].length;
  const w = useMemo(() => makeRnnWeights(hidden, I, rho), [hidden, I, rho]);
  const steps = useMemo(() => runRnn(w, input.xs), [w, input.xs]);
  const probe = useMemo(() => memoryProbe(w, input.xs, eps), [w, input.xs, eps]);

  const T = steps.length;
  const total = T * RNN_SUBSTEPS.length;
  const s = useStepper(total, { baseMs: 900 });
  const { reset } = s;
  useEffect(() => {
    reset();
  }, [T, hidden, mode, reset]);

  const cur = Math.min(s.step, total - 1);
  const t = Math.floor(cur / RNN_SUBSTEPS.length);
  const subIdx = cur % RNN_SUBSTEPS.length;
  const sub = RNN_SUBSTEPS[subIdx];
  const st = steps[t];

  let eq = "";
  let lines: string[] = [];
  let explain = "";
  if (sub === "input") {
    eq = "W_xh · x_t";
    lines = [`unit 1: ${expandRow(w.Wxh, st.x, 0, st.xin[0])}`, `W_xh·x_t = ${vec(st.xin)}`];
    explain = mode === "chars" ? `x_t is the one-hot vector of "${input.labels[t].replace(/"/g, "")}" over the vocabulary [${input.vocab.join(", ")}]. It is all zeros except one 1, so W_xh·x_t just picks out one column of W_xh.` : "The input is mapped into the hidden space. The same W_xh is used at every time step.";
  } else if (sub === "recurrent") {
    eq = "W_hh · h_{t−1}";
    lines = [`h_{t−1} = ${vec(st.hPrev)}`, `unit 1: ${expandRow(w.Whh, st.hPrev, 0, st.rec[0])}`, `W_hh·h_{t−1} = ${vec(st.rec)}`];
    explain = t === 0 ? "At t = 1 the previous state is h₀ = 0, so the recurrent term is exactly zero." : "This is the memory path: the previous state, mixed by the recurrent matrix. How big it is next to the input term decides how much the past matters.";
  } else if (sub === "sum") {
    eq = "z_t = W_xh·x_t + W_hh·h_{t−1} + b";
    lines = [`z_t = ${vec(st.xin)} + ${vec(st.rec)} + ${vec(w.b)}`, `z_t = ${vec(st.z)}`];
    explain = "Add the two terms and the bias. Old memory and new input now live in one vector.";
  } else if (sub === "tanh") {
    eq = "h_t = tanh(z_t)";
    lines = [`h_t = tanh(${vec(st.z)})`, `h_t = ${vec(st.h)}`, `1 − h_t² = ${vec(st.h.map((v) => 1 - v * v))}`];
    explain = "tanh squashes every unit into the range −1 to 1. Units near ±1 are saturated (1 − h² ≈ 0), so they barely react to changes. This matters a lot when training with backpropagation through time.";
  } else {
    eq = "y_t = W_hy · h_t + b_y";
    lines = [`y_t = ${st.h.map((v, k) => `${fmt(w.Why[k], 2)}×${fmt(v, 2)}`).join(" + ")} + ${fmt(w.by, 2)}`, `y_t = ${fmt(st.y, 4)}`];
    explain = t + 1 < T ? `h_t is passed on to step ${t + 2}. The output head reads it at every step; use that output or ignore it, depending on the task.` : "Last step. In a many-to-one task (like judging a review as positive or negative), only this y_T would be used.";
  }

  const rows = [
    { name: "x_t", v: st.x, on: sub === "input", vis: true, max: 1.5 },
    { name: "h_{t−1}", v: st.hPrev, on: sub === "recurrent", vis: true, max: 1 },
    { name: "W_xh·x_t", v: st.xin, on: sub === "input" || sub === "sum", vis: subIdx >= 0, max: 2 },
    { name: "W_hh·h_{t−1}", v: st.rec, on: sub === "recurrent" || sub === "sum", vis: subIdx >= 1, max: 2 },
    { name: "b", v: w.b, on: sub === "sum", vis: subIdx >= 2, max: 2 },
    { name: "z_t", v: st.z, on: sub === "sum" || sub === "tanh", vis: subIdx >= 2, max: 2 },
    { name: "h_t", v: st.h, on: sub === "tanh" || sub === "output", vis: subIdx >= 3, max: 1 },
    { name: "y_t", v: [st.y], on: sub === "output", vis: subIdx >= 4, max: 1.5 },
  ];

  const last = probe.dh[T - 1];

  return (
    <SimShell
      title="Vanilla RNN, Unrolled Through Time"
      subtitle={`Input size ${I}, hidden size ${hidden}, fixed random weights. W_hh is scaled to spectral radius ${fmt(rho, 2)}. Every number is computed live.`}
      controls={
        <div className="space-y-2">
          <Controls s={s} />
          <div className="grid gap-2 sm:grid-cols-2">
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="w-24 shrink-0 text-muted">Input type</span>
              <button className={`sim-btn ${mode === "numbers" ? "primary" : ""}`} onClick={() => setMode("numbers")} aria-label="Numeric input sequence">
                Numbers
              </button>
              <button className={`sim-btn ${mode === "chars" ? "primary" : ""}`} onClick={() => setMode("chars")} aria-label="Character input sequence (one-hot)">
                Characters
              </button>
            </div>
            <label className="flex items-center gap-2 text-xs">
              <span className="w-24 shrink-0 text-muted">{mode === "numbers" ? "Sequence x" : "Text (≤4 letters)"}</span>
              {mode === "numbers" ? (
                <input className="min-w-0 flex-1 rounded border border-border bg-surface px-2 py-1 font-mono text-xs" value={numText} onChange={(e) => setNumText(e.target.value)} aria-label="Input sequence, comma separated numbers" />
              ) : (
                <input className="min-w-0 flex-1 rounded border border-border bg-surface px-2 py-1 font-mono text-xs" value={charText} onChange={(e) => setCharText(e.target.value)} aria-label="Input text, characters become one-hot vectors" />
              )}
            </label>
            <Slider label="Spectral radius ρ" value={rho} min={0.1} max={2.5} step={0.05} onChange={setRho} fmt={(v) => v.toFixed(2)} />
            <Slider label="Probe ε on x₁" value={eps} min={0.01} max={1} step={0.01} onChange={setEps} fmt={(v) => v.toFixed(2)} />
            <div className="flex items-center gap-2 text-xs">
              <span className="w-24 shrink-0 text-muted">Hidden size</span>
              {[3, 4].map((n) => (
                <button key={n} className={`sim-btn ${hidden === n ? "primary" : ""}`} onClick={() => setHidden(n)} aria-label={`Hidden size ${n}`}>
                  {n}
                </button>
              ))}
            </div>
          </div>
        </div>
      }
      aside={
        <div className="space-y-3">
          <Narration title={`t = ${t + 1}, x_t = ${input.labels[t]} — ${SUB_LABEL[sub]}`}>
            <p className="mb-2">{explain}</p>
            <div className="font-mono text-[11px] leading-relaxed text-text">
              <div className="mb-1 font-semibold">{eq}</div>
              {lines.map((l, k) => (
                <div key={k} className="break-words">
                  {l}
                </div>
              ))}
            </div>
          </Narration>
          <div className="text-xs text-muted">
            Try this: keep the default sequence and move ρ from 0.3 to 1.5 while watching the memory test. With small ρ, the effect of x₁ fades in a few steps. With large ρ, it lasts much longer. Then switch to Characters and type a word with up to four different letters.
          </div>
        </div>
      }
    >
      <div className="overflow-x-auto">
        <Diagram steps={steps} t={t} sub={sub} labels={input.labels} w={w} />
      </div>

      <div className="mt-3 space-y-1">
        {rows.map((r) => (
          <VecRow key={r.name} name={r.name} v={r.v} on={r.on} visible={r.vis} maxAbs={r.max} />
        ))}
      </div>

      <div className="mt-4 text-xs text-muted">
        Memory test: we change x₁ by ε and run the whole sequence again. Bars show ‖h_t(perturbed) − h_t‖ on a log scale: how much of the first input each hidden state still carries.
      </div>
      <div className="overflow-x-auto">
        <ProbeChart dh={probe.dh} eps={eps} />
      </div>
      <div className="mt-1 font-mono text-[11px] text-muted">
        ‖Δh_T‖ / ε = {sci(last / eps)} · |Δy_T| = {sci(probe.dy[T - 1])} · ρ(W_hh) = {fmt(rho, 2)}
      </div>
    </SimShell>
  );
}
