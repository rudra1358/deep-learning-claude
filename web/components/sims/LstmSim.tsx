"use client";

import { useEffect, useMemo, useState } from "react";
import { Controls, Narration, SimShell, Slider, fmt, signedColor, useStepper } from "./SimShell";
import { SUBSTEPS, makeLstmWeights, parseSequence, runLstm, withForgetBias, type GateName, type LstmStep, type LstmWeights } from "@/lib/sim/lstm";

const SUB_LABEL: Record<(typeof SUBSTEPS)[number], string> = {
  forget: "1 · Forget gate",
  input: "2 · Input gate & candidate",
  cell: "3 · Cell-state update",
  output: "4 · Output gate",
  hidden: "5 · Hidden state",
};

const vec = (v: number[], d = 3) => "[" + v.map((x) => fmt(x, d)).join(", ") + "]";

/** Expand unit k of gate `gate`: W·[h, x] + b with every number substituted. */
function expand(w: LstmWeights, gate: GateName, s: LstmStep, k: number) {
  const z = [...s.hPrev, ...s.x];
  const terms = w.W[gate][k].map((wv, j) => `${fmt(wv, 2)}×${fmt(z[j], 2)}`);
  return `${terms.join(" + ")} + ${fmt(w.b[gate][k], 2)} = ${fmt(s.pre[gate][k], 3)}`;
}

/** A tiny group of signed bars (one per hidden unit) drawn around baseline y. */
function Bars({ x, y, values, label, show = true }: { x: number; y: number; values: number[]; label?: string; show?: boolean }) {
  const bw = 7, gap = 3, scale = 16;
  return (
    <g>
      {label && (
        <text x={x} y={y - scale - 5} fontSize={10} fill="var(--text-muted)">
          {label}
        </text>
      )}
      <line x1={x - 2} x2={x + values.length * (bw + gap)} y1={y} y2={y} stroke="var(--border)" />
      {show &&
        values.map((v, k) => {
          const hgt = Math.min(1, Math.abs(v)) * scale;
          return <rect key={k} x={x + k * (bw + gap)} y={v >= 0 ? y - hgt : y} width={bw} height={Math.max(0.5, hgt)} fill={v >= 0 ? "var(--accent)" : "var(--clay)"} />;
        })}
    </g>
  );
}

export default function LstmSim() {
  const [seqText, setSeqText] = useState("1, 0, 0, 0, 0, 1, 0, 0");
  const [bf, setBf] = useState(1);
  const [hidden, setHidden] = useState(3);

  const seq = useMemo(() => {
    const p = parseSequence(seqText);
    return p.length ? p : [0];
  }, [seqText]);
  const base = useMemo(() => makeLstmWeights(hidden, 1), [hidden]);
  const w = useMemo(() => withForgetBias(base, bf), [base, bf]);
  const steps = useMemo(() => runLstm(w, seq), [w, seq]);

  const total = seq.length * SUBSTEPS.length;
  const s = useStepper(total, { baseMs: 1100 });
  const { reset } = s;
  useEffect(() => {
    reset();
  }, [seq.length, hidden, reset]);

  const cur = Math.min(s.step, total - 1);
  const t = Math.floor(cur / SUBSTEPS.length);
  const subIdx = cur % SUBSTEPS.length;
  const sub = SUBSTEPS[subIdx];
  const st = steps[t];

  const shown = { f: subIdx >= 0, ig: subIdx >= 1, c: subIdx >= 2, o: subIdx >= 3, h: subIdx >= 4 };
  const active = (k: (typeof SUBSTEPS)[number]) => sub === k;
  const stroke = (on: boolean) => (on ? "var(--gold)" : "var(--text-muted)");
  const sw = (on: boolean) => (on ? 2.6 : 1.4);
  const gateFill = (on: boolean) => (on ? "color-mix(in srgb, var(--gold) 22%, var(--surface))" : "var(--surface)");

  const meanF = useMemo(() => {
    const fs = steps.slice(0, t + 1).flatMap((x) => x.f);
    return fs.reduce((a, b) => a + b, 0) / fs.length;
  }, [steps, t]);

  // ---------- Narration text (numbers substituted) ----------
  let eqTitle = "";
  let eqLines: string[] = [];
  let explain = "";
  if (sub === "forget") {
    eqTitle = "f_t = σ(W_f·[h_{t−1}, x_t] + b_f)";
    eqLines = [`unit 1 pre-activation: ${expand(w, "f", st, 0)}`, `f_t = σ(${vec(st.pre.f)}) = ${vec(st.f)}`, `kept memory f_t ⊙ c_{t−1} = ${vec(st.f)} ⊙ ${vec(st.cPrev)} = ${vec(st.kept)}`];
    explain = `The forget gate decides how much of each unit's old memory to keep (0 = forget all, 1 = keep all). With b_f = ${fmt(bf, 1)}, the average so far is ${fmt(meanF, 2)}. Raise b_f and memories last longer.`;
  } else if (sub === "input") {
    eqTitle = "i_t = σ(W_i·[h_{t−1}, x_t] + b_i),  g_t = tanh(W_g·[h_{t−1}, x_t] + b_g)";
    eqLines = [`unit 1 (input): ${expand(w, "i", st, 0)}`, `i_t = ${vec(st.i)}`, `unit 1 (candidate): ${expand(w, "g", st, 0)}`, `g_t = tanh(${vec(st.pre.g)}) = ${vec(st.g)}`, `written i_t ⊙ g_t = ${vec(st.written)}`];
    explain = "The candidate g_t suggests new content, between −1 and 1. The input gate i_t decides, for each unit, how much of it really gets written.";
  } else if (sub === "cell") {
    eqTitle = "c_t = f_t ⊙ c_{t−1} + i_t ⊙ g_t";
    eqLines = [`c_t = ${vec(st.kept)} + ${vec(st.written)}`, `c_t = ${vec(st.c)}`];
    explain = "The cell state (the long-term memory) only gets scaled by f_t and then added to. There is no weight matrix and no squashing on this path. That is why information and gradients can travel along it for many steps.";
  } else if (sub === "output") {
    eqTitle = "o_t = σ(W_o·[h_{t−1}, x_t] + b_o)";
    eqLines = [`unit 1: ${expand(w, "o", st, 0)}`, `o_t = σ(${vec(st.pre.o)}) = ${vec(st.o)}`];
    explain = "The output gate decides which parts of the memory the rest of the network gets to see at this step. A unit can hold a large c_t but show almost nothing in h_t.";
  } else {
    eqTitle = "h_t = o_t ⊙ tanh(c_t)";
    eqLines = [`tanh(c_t) = tanh(${vec(st.c)}) = ${vec(st.tanhC)}`, `h_t = ${vec(st.o)} ⊙ ${vec(st.tanhC)} = ${vec(st.h)}`];
    explain = t + 1 < seq.length ? `h_t and c_t are passed on to step ${t + 2}. h_t is also the layer's output at this step.` : "This was the last step. Usually h_T now goes to a final layer that makes the prediction.";
  }

  // ---------- Vector panel rows ----------
  const rows: { name: string; v: number[]; visible: boolean; on: boolean }[] = [
    { name: "x_t", v: st.x, visible: true, on: false },
    { name: "h_{t−1}", v: st.hPrev, visible: true, on: sub === "forget" || sub === "input" || sub === "output" },
    { name: "c_{t−1}", v: st.cPrev, visible: true, on: sub === "forget" },
    { name: "f_t", v: st.f, visible: shown.f, on: sub === "forget" || sub === "cell" },
    { name: "i_t", v: st.i, visible: shown.ig, on: sub === "input" || sub === "cell" },
    { name: "g_t", v: st.g, visible: shown.ig, on: sub === "input" || sub === "cell" },
    { name: "c_t", v: st.c, visible: shown.c, on: sub === "cell" || sub === "hidden" },
    { name: "o_t", v: st.o, visible: shown.o, on: sub === "output" || sub === "hidden" },
    { name: "h_t", v: st.h, visible: shown.h, on: sub === "hidden" },
  ];

  // ---------- Timeline geometry ----------
  const colW = 50, labelW = 46, cellH = 18;
  const tlRows = [{ label: "x", get: (q: LstmStep) => q.x, kind: "x" as const }, ...Array.from({ length: hidden }, (_, k) => ({ label: `c${k + 1}`, get: (q: LstmStep) => [q.c[k]], kind: "c" as const })), ...Array.from({ length: hidden }, (_, k) => ({ label: `h${k + 1}`, get: (q: LstmStep) => [q.h[k]], kind: "h" as const }))];
  const tlW = labelW + seq.length * colW + 4;
  const tlH = 16 + tlRows.length * (cellH + 2) + 4;
  const visibleAt = (tau: number, kind: "x" | "c" | "h") => tau < t || (tau === t && (kind === "x" || (kind === "c" && shown.c) || (kind === "h" && shown.h)));

  return (
    <SimShell
      title="LSTM Cell, Step by Step"
      subtitle={`Input size 1, hidden size ${hidden}, fixed random weights. Every number is computed live.`}
      controls={
        <div className="space-y-2">
          <Controls s={s} />
          <div className="grid gap-2 sm:grid-cols-2">
            <label className="flex items-center gap-2 text-xs">
              <span className="w-24 shrink-0 text-muted">Sequence x</span>
              <input className="min-w-0 flex-1 rounded border border-border bg-surface px-2 py-1 font-mono text-xs" value={seqText} onChange={(e) => setSeqText(e.target.value)} aria-label="Input sequence, comma separated" />
            </label>
            <Slider label="Forget bias b_f" value={bf} min={-3} max={5} step={0.1} onChange={setBf} fmt={(v) => v.toFixed(1)} />
            <label className="flex items-center gap-2 text-xs">
              <span className="w-24 shrink-0 text-muted">Hidden size</span>
              {[3, 4].map((n) => (
                <button key={n} className={`sim-btn ${hidden === n ? "primary" : ""}`} onClick={() => setHidden(n)} aria-label={`Hidden size ${n}`}>
                  {n}
                </button>
              ))}
            </label>
          </div>
        </div>
      }
      aside={
        <div className="space-y-3">
          <Narration title={`t = ${t + 1}, x_t = ${fmt(st.x[0], 2)} — ${SUB_LABEL[sub]}`}>
            <p className="mb-2">{explain}</p>
            <div className="font-mono text-[11px] leading-relaxed text-text">
              <div className="mb-1 font-semibold">{eqTitle}</div>
              {eqLines.map((l, k) => (
                <div key={k} className="break-words">
                  {l}
                </div>
              ))}
            </div>
          </Narration>
          <div className="text-xs text-muted">
            Try this: play the default sequence with b_f = −2. The pulse at t = 1 vanishes from c_t within two steps. Set b_f = +4 and most of it is still there at t = 5. Then edit the sequence (values from −3 to 3, up to 12 steps).
          </div>
        </div>
      }
    >
      <svg viewBox="0 0 720 300" className="h-auto w-full" role="img" aria-label={`LSTM cell diagram at time step ${t + 1}, ${SUB_LABEL[sub]}`}>
        <rect x={70} y={38} width={530} height={244} rx={14} fill="var(--surface-2)" stroke="var(--border)" />
        {/* cell-state conveyor belt */}
        <line x1={14} x2={706} y1={75} y2={75} stroke={stroke(sub === "forget" || sub === "cell")} strokeWidth={sub === "cell" ? 3.4 : 2.4} />
        <text x={14} y={66} fontSize={12} fill="var(--text)">c_(t−1)</text>
        <text x={664} y={66} fontSize={12} fill="var(--text)">c_t</text>
        <Bars x={18} y={110} values={st.cPrev} label="c_(t−1)" />
        <Bars x={650} y={110} values={st.c} label="c_t" show={shown.c} />

        {/* hidden-state bus and input */}
        <line x1={14} x2={440} y1={262} y2={262} stroke={stroke(sub === "forget" || sub === "input" || sub === "output")} strokeWidth={1.4} />
        <text x={14} y={254} fontSize={12} fill="var(--text)">h_(t−1)</text>
        <Bars x={18} y={240} values={st.hPrev} />
        <line x1={110} x2={110} y1={298} y2={262} stroke="var(--text-muted)" strokeWidth={1.4} />
        <text x={118} y={294} fontSize={12} fill="var(--text)">x_t = {fmt(st.x[0], 2)}</text>

        {/* gate stubs from the bus */}
        {[180, 270, 340, 440].map((gx) => (
          <line key={gx} x1={gx} x2={gx} y1={262} y2={220} stroke="var(--text-muted)" strokeWidth={1.2} />
        ))}

        {/* forget path */}
        <line x1={180} x2={180} y1={190} y2={87} stroke={stroke(active("forget"))} strokeWidth={sw(active("forget"))} />
        {/* input path */}
        <polyline points="270,190 270,140 328,140" fill="none" stroke={stroke(active("input"))} strokeWidth={sw(active("input"))} />
        <line x1={340} x2={340} y1={190} y2={152} stroke={stroke(active("input"))} strokeWidth={sw(active("input"))} />
        <line x1={340} x2={340} y1={128} y2={87} stroke={stroke(active("input") || active("cell"))} strokeWidth={sw(active("cell"))} />
        {/* output path */}
        <line x1={540} x2={540} y1={75} y2={112} stroke={stroke(active("hidden"))} strokeWidth={sw(active("hidden"))} />
        <line x1={540} x2={540} y1={138} y2={173} stroke={stroke(active("hidden"))} strokeWidth={sw(active("hidden"))} />
        <polyline points="440,190 440,185 528,185" fill="none" stroke={stroke(active("output"))} strokeWidth={sw(active("output"))} />
        <polyline points="540,197 540,262 706,262" fill="none" stroke={stroke(active("hidden"))} strokeWidth={sw(active("hidden"))} />
        <text x={664} y={254} fontSize={12} fill="var(--text)">h_t</text>
        <Bars x={650} y={240} values={st.h} show={shown.h} />

        {/* pointwise nodes */}
        {[
          { x: 180, y: 75, sym: "×", on: active("forget") },
          { x: 340, y: 75, sym: "+", on: active("cell") },
          { x: 340, y: 140, sym: "×", on: active("input") },
          { x: 540, y: 185, sym: "×", on: active("hidden") },
        ].map((n, k) => (
          <g key={k}>
            <circle cx={n.x} cy={n.y} r={12} fill={gateFill(n.on)} stroke={stroke(n.on)} strokeWidth={sw(n.on)} />
            <text x={n.x} y={n.y + 5} textAnchor="middle" fontSize={15} fill="var(--text)">
              {n.sym}
            </text>
          </g>
        ))}
        <rect x={518} y={112} width={44} height={26} rx={13} fill={gateFill(active("hidden"))} stroke={stroke(active("hidden"))} strokeWidth={sw(active("hidden"))} />
        <text x={540} y={129} textAnchor="middle" fontSize={11} fill="var(--text)">tanh</text>

        {/* gate blocks */}
        {[
          { x: 180, sym: "σ", name: "f_t", v: st.f, on: active("forget"), show: shown.f },
          { x: 270, sym: "σ", name: "i_t", v: st.i, on: active("input"), show: shown.ig },
          { x: 340, sym: "tanh", name: "g_t", v: st.g, on: active("input"), show: shown.ig },
          { x: 440, sym: "σ", name: "o_t", v: st.o, on: active("output"), show: shown.o },
        ].map((g) => (
          <g key={g.name}>
            <rect x={g.x - 23} y={190} width={46} height={30} rx={5} fill={gateFill(g.on)} stroke={stroke(g.on)} strokeWidth={sw(g.on)} />
            <text x={g.x} y={210} textAnchor="middle" fontSize={13} fill="var(--text)">
              {g.sym}
            </text>
            <text x={g.x - 26} y={186} textAnchor="end" fontSize={11} fill="var(--text-muted)">
              {g.name}
            </text>
            <Bars x={g.x + 6} y={180} values={g.v} show={g.show} />
          </g>
        ))}
      </svg>

      {/* vector values */}
      <div className="mt-3 grid grid-cols-1 gap-x-6 gap-y-1 font-mono text-xs sm:grid-cols-2">
        {rows.map((r) => (
          <div key={r.name} className={`flex items-center gap-2 rounded px-1.5 py-0.5 ${r.on ? "bg-surface-2" : ""}`} style={r.on ? { outline: "1px solid var(--gold)" } : undefined}>
            <span className="w-14 shrink-0 text-muted">{r.name}</span>
            {r.visible ? (
              <span className="flex flex-wrap gap-1">
                {r.v.map((v, k) => (
                  <span key={k} className="rounded px-1" style={{ background: signedColor(v, 1) }}>
                    {fmt(v, 3)}
                  </span>
                ))}
              </span>
            ) : (
              <span className="text-faint">not computed yet</span>
            )}
          </div>
        ))}
      </div>

      {/* timeline */}
      <div className="mt-4 text-xs text-muted">Timeline: the input, cell state c_t and hidden state h_t at each step so far (one column per step).</div>
      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${tlW} ${tlH}`} className="h-auto w-full" style={{ minWidth: Math.min(tlW, 520) }} role="img" aria-label="Timeline of cell state and hidden state across time steps">
          {Array.from({ length: seq.length }, (_, tau) => (
            <text key={tau} x={labelW + tau * colW + colW / 2} y={11} textAnchor="middle" fontSize={10} fill={tau === t ? "var(--gold)" : "var(--text-muted)"}>
              t={tau + 1}
            </text>
          ))}
          {tlRows.map((row, r) => {
            const y = 16 + r * (cellH + 2);
            return (
              <g key={row.label}>
                <text x={labelW - 6} y={y + 13} textAnchor="end" fontSize={10} fill="var(--text-muted)">
                  {row.label}
                </text>
                {steps.map((q, tau) => {
                  const vis = visibleAt(tau, row.kind);
                  const v = row.get(q)[0];
                  return (
                    <g key={tau}>
                      <rect x={labelW + tau * colW + 1} y={y} width={colW - 2} height={cellH} rx={3} fill={vis ? signedColor(v, 1) : "var(--surface-2)"} stroke={tau === t ? "var(--gold)" : "var(--border)"} strokeWidth={tau === t ? 1.5 : 0.6} />
                      {vis && (
                        <text x={labelW + tau * colW + colW / 2} y={y + 13} textAnchor="middle" fontSize={10} fill="var(--text)">
                          {fmt(v, 2)}
                        </text>
                      )}
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
      </div>
      <div className="mt-1 font-mono text-[11px] text-muted">average forget gate over steps 1..{t + 1}: {fmt(meanF, 3)}</div>
    </SimShell>
  );
}
