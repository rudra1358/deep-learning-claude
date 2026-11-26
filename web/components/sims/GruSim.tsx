"use client";

import { useEffect, useMemo, useState } from "react";
import { Controls, Narration, SimShell, Slider, fmt, signedColor, useStepper } from "./SimShell";
import { GRU_SUBSTEPS, makeGruWeights, makeRnnWeights, memoryProbe, parseSequence, runGru, withUpdateBias, type GruStep, type GruSubStep } from "@/lib/sim/gru";

const SUB_LABEL: Record<GruSubStep, string> = {
  reset: "1 · Reset gate",
  candidate: "2 · Candidate",
  update: "3 · Update gate",
  hidden: "4 · New hidden state",
};

const vec = (v: number[], d = 3) => "[" + v.map((x) => fmt(x, d)).join(", ") + "]";

/** Expand one unit: Σ w·v + b = pre, with every number substituted. */
function expand(row: number[], inputs: number[], b: number, pre: number) {
  const terms = row.map((wv, j) => `${fmt(wv, 2)}×${fmt(inputs[j], 2)}`);
  return `${terms.join(" + ")} + ${fmt(b, 2)} = ${fmt(pre, 3)}`;
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

export default function GruSim() {
  const [seqText, setSeqText] = useState("1, 0, 0, 0, 0, 0, 0, 0, 0, 0");
  const [bz, setBz] = useState(-1.5);
  const [hidden, setHidden] = useState(3);

  const seq = useMemo(() => {
    const p = parseSequence(seqText);
    return p.length ? p : [0];
  }, [seqText]);
  const base = useMemo(() => makeGruWeights(hidden), [hidden]);
  const w = useMemo(() => withUpdateBias(base, bz), [base, bz]);
  const rnnW = useMemo(() => makeRnnWeights(hidden), [hidden]);
  const steps = useMemo(() => runGru(w, seq), [w, seq]);
  const probe = useMemo(() => memoryProbe(w, rnnW, seq, 1), [w, rnnW, seq]);

  const total = seq.length * GRU_SUBSTEPS.length;
  const s = useStepper(total, { baseMs: 1100 });
  const { reset } = s;
  useEffect(() => {
    reset();
  }, [seq.length, hidden, reset]);

  const cur = Math.min(s.step, total - 1);
  const t = Math.floor(cur / GRU_SUBSTEPS.length);
  const subIdx = cur % GRU_SUBSTEPS.length;
  const sub = GRU_SUBSTEPS[subIdx];
  const st = steps[t];

  const shown = { r: subIdx >= 0, c: subIdx >= 1, z: subIdx >= 2, h: subIdx >= 3 };
  const active = (k: GruSubStep) => sub === k;
  const stroke = (on: boolean) => (on ? "var(--gold)" : "var(--text-muted)");
  const sw = (on: boolean) => (on ? 2.6 : 1.4);
  const gateFill = (on: boolean) => (on ? "color-mix(in srgb, var(--gold) 22%, var(--surface))" : "var(--surface)");

  const meanZ = useMemo(() => {
    const zs = steps.slice(0, t + 1).flatMap((q) => q.z);
    return zs.reduce((a, b) => a + b, 0) / zs.length;
  }, [steps, t]);

  // ---------- Narration (numbers substituted) ----------
  const zin = [...st.hPrev, ...st.x];
  let eqTitle = "";
  let eqLines: string[] = [];
  let explain = "";
  if (sub === "reset") {
    eqTitle = "r_t = σ(W_r·[h_{t−1}, x_t] + b_r)";
    eqLines = [`unit 1: ${expand(w.W.r[0], zin, w.b.r[0], st.pre.r[0])}`, `r_t = σ(${vec(st.pre.r)}) = ${vec(st.r)}`, `r_t ⊙ h_{t−1} = ${vec(st.r)} ⊙ ${vec(st.hPrev)} = ${vec(st.rh)}`];
    explain = "The reset gate decides, for each unit, how much of the old memory the candidate may look at. If r ≈ 0, the candidate ignores the past and is built almost only from x_t.";
  } else if (sub === "candidate") {
    eqTitle = "h̃_t = tanh(W_h·[r_t ⊙ h_{t−1}, x_t] + b_h)";
    eqLines = [`unit 1: ${expand(w.W.h[0], [...st.rh, ...st.x], w.b.h[0], st.pre.h[0])}`, `h̃_t = tanh(${vec(st.pre.h)}) = ${vec(st.htilde)}`];
    explain = "The candidate is a suggested new state, between −1 and 1, made the same way a plain RNN would. It sees the old state only through the reset gate.";
  } else if (sub === "update") {
    eqTitle = "z_t = σ(W_z·[h_{t−1}, x_t] + b_z)";
    eqLines = [`unit 1: ${expand(w.W.z[0], zin, w.b.z[0], st.pre.z[0])}`, `z_t = σ(${vec(st.pre.z)}) = ${vec(st.z)}`, `1 − z_t = ${vec(st.z.map((v) => 1 - v))}`];
    explain = `z is how much of each unit gets rewritten at this step. With b_z = ${fmt(bz, 1)}, the average z so far is ${fmt(meanZ, 2)}. Lower b_z means smaller z, so the memory is carried forward almost unchanged.`;
  } else {
    eqTitle = "h_t = (1 − z_t) ⊙ h_{t−1} + z_t ⊙ h̃_t";
    eqLines = [`kept (1 − z_t) ⊙ h_{t−1} = ${vec(st.keep)}`, `written z_t ⊙ h̃_t = ${vec(st.write)}`, `h_t = ${vec(st.h)}`];
    explain =
      t + 1 < seq.length
        ? `Each unit blends its old state with the candidate. The (1 − z) path has no matrix, so when z ≈ 0 the old state passes straight through. h_t now goes to step ${t + 2}.`
        : "This is the last step. Normally h_T would now go to a final layer that makes the prediction.";
  }

  // ---------- Vector panel rows ----------
  const rows: { name: string; v: number[]; visible: boolean; on: boolean }[] = [
    { name: "x_t", v: st.x, visible: true, on: false },
    { name: "h_{t−1}", v: st.hPrev, visible: true, on: sub === "reset" || sub === "update" || sub === "hidden" },
    { name: "r_t", v: st.r, visible: shown.r, on: sub === "reset" },
    { name: "r⊙h_{t−1}", v: st.rh, visible: shown.r, on: sub === "reset" || sub === "candidate" },
    { name: "h̃_t", v: st.htilde, visible: shown.c, on: sub === "candidate" || sub === "hidden" },
    { name: "z_t", v: st.z, visible: shown.z, on: sub === "update" || sub === "hidden" },
    { name: "h_t", v: st.h, visible: shown.h, on: sub === "hidden" },
  ];

  // ---------- Timeline geometry ----------
  const colW = 50, labelW = 46, cellH = 18;
  const tlRows = [
    { label: "x", get: (q: GruStep) => q.x[0], kind: "x" as const },
    ...Array.from({ length: hidden }, (_, k) => ({ label: `z${k + 1}`, get: (q: GruStep) => q.z[k], kind: "z" as const })),
    ...Array.from({ length: hidden }, (_, k) => ({ label: `h${k + 1}`, get: (q: GruStep) => q.h[k], kind: "h" as const })),
  ];
  const tlW = labelW + seq.length * colW + 4;
  const tlH = 16 + tlRows.length * (cellH + 2) + 4;
  const visibleAt = (tau: number, kind: "x" | "z" | "h") => tau < t || (tau === t && (kind === "x" || (kind === "z" && shown.z) || (kind === "h" && shown.h)));

  // ---------- Memory-probe geometry ----------
  const pH = 46;
  const pMax = Math.max(1e-9, ...probe.gru, ...probe.rnn);
  const pRows = [
    { label: "GRU", v: probe.gru, color: "var(--accent)" },
    { label: "RNN", v: probe.rnn, color: "var(--clay)" },
  ];
  const pSvgH = 16 + pRows.length * (pH + 18);
  const ratio = (v: number[]) => (v[0] > 1e-9 ? v[Math.min(t, v.length - 1)] / v[0] : 0);

  return (
    <SimShell
      title="GRU Cell, Step by Step"
      subtitle={`Input size 1, hidden size ${hidden}, fixed random weights. We compare it with a plain RNN of the same size. Every number is computed live.`}
      controls={
        <div className="space-y-2">
          <Controls s={s} />
          <div className="grid gap-2 sm:grid-cols-2">
            <label className="flex items-center gap-2 text-xs">
              <span className="w-24 shrink-0 text-muted">Sequence x</span>
              <input className="min-w-0 flex-1 rounded border border-border bg-surface px-2 py-1 font-mono text-xs" value={seqText} onChange={(e) => setSeqText(e.target.value)} aria-label="Input sequence, comma separated" />
            </label>
            <Slider label="Update bias b_z" value={bz} min={-4} max={4} step={0.1} onChange={setBz} fmt={(v) => v.toFixed(1)} />
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
            Try this: with the default sequence, set b_z = −3. The GRU&apos;s memory bars stay almost level for all 10 steps. Set b_z = +2 and they shrink faster than the plain RNN&apos;s. Values from −3 to 3, up to 12 steps.
          </div>
        </div>
      }
    >
      <svg viewBox="0 0 720 300" className="h-auto w-full" role="img" aria-label={`GRU cell diagram at time step ${t + 1}, ${SUB_LABEL[sub]}`}>
        <rect x={70} y={38} width={530} height={244} rx={14} fill="var(--surface-2)" stroke="var(--border)" />

        {/* state line h_{t-1} -> h_t */}
        <line x1={14} x2={288} y1={75} y2={75} stroke={stroke(active("reset") || active("hidden"))} strokeWidth={2.4} />
        <line x1={312} x2={508} y1={75} y2={75} stroke={stroke(active("hidden"))} strokeWidth={active("hidden") ? 3.4 : 2.4} />
        <line x1={532} x2={706} y1={75} y2={75} stroke={stroke(active("hidden"))} strokeWidth={active("hidden") ? 3.4 : 2.4} />
        <text x={14} y={66} fontSize={12} fill="var(--text)">h_(t−1)</text>
        <text x={672} y={66} fontSize={12} fill="var(--text)">h_t</text>
        <Bars x={18} y={110} values={st.hPrev} label="h_(t−1)" />
        <Bars x={650} y={110} values={st.h} label="h_t" show={shown.h} />

        {/* input bus: h_{t-1} and x_t feed every gate */}
        <line x1={14} x2={440} y1={262} y2={262} stroke={stroke(active("reset") || active("update"))} strokeWidth={1.4} />
        <text x={14} y={254} fontSize={12} fill="var(--text)">h_(t−1)</text>
        <Bars x={18} y={240} values={st.hPrev} />
        <line x1={110} x2={110} y1={298} y2={262} stroke="var(--text-muted)" strokeWidth={1.4} />
        <text x={118} y={294} fontSize={12} fill="var(--text)">x_t = {fmt(st.x[0], 2)}</text>
        {[170, 300, 440].map((gx) => (
          <line key={gx} x1={gx} x2={gx} y1={262} y2={220} stroke="var(--text-muted)" strokeWidth={1.2} />
        ))}

        {/* reset path: r -> (×) with h_{t-1} from the state line */}
        <polyline points="170,190 170,140 218,140" fill="none" stroke={stroke(active("reset"))} strokeWidth={sw(active("reset"))} />
        <line x1={230} x2={230} y1={75} y2={128} stroke={stroke(active("reset"))} strokeWidth={sw(active("reset"))} />
        {/* r ⊙ h -> candidate */}
        <polyline points="242,140 380,140 380,205 417,205" fill="none" stroke={stroke(active("candidate"))} strokeWidth={sw(active("candidate"))} />
        {/* update gate -> (1−z)× node and z× node */}
        <line x1={300} x2={300} y1={190} y2={87} stroke={stroke(active("update") || active("hidden"))} strokeWidth={sw(active("update"))} />
        <polyline points="300,165 520,165 520,152" fill="none" stroke={stroke(active("update") || active("hidden"))} strokeWidth={sw(active("update"))} />
        {/* candidate -> z× node -> + */}
        <polyline points="440,190 440,140 508,140" fill="none" stroke={stroke(active("candidate") || active("hidden"))} strokeWidth={sw(active("candidate"))} />
        <line x1={520} x2={520} y1={128} y2={87} stroke={stroke(active("hidden"))} strokeWidth={sw(active("hidden"))} />

        {/* pointwise nodes */}
        {[
          { x: 230, y: 140, sym: "×", on: active("reset"), tag: "r⊙h" },
          { x: 300, y: 75, sym: "×", on: active("hidden"), tag: "1−z" },
          { x: 520, y: 140, sym: "×", on: active("hidden"), tag: "z" },
          { x: 520, y: 75, sym: "+", on: active("hidden"), tag: "" },
        ].map((n, k) => (
          <g key={k}>
            <circle cx={n.x} cy={n.y} r={12} fill={gateFill(n.on)} stroke={stroke(n.on)} strokeWidth={sw(n.on)} />
            <text x={n.x} y={n.y + 5} textAnchor="middle" fontSize={15} fill="var(--text)">
              {n.sym}
            </text>
            {n.tag && (
              <text x={n.x + 16} y={n.y - 10} fontSize={10} fill="var(--text-muted)">
                {n.tag}
              </text>
            )}
          </g>
        ))}

        {/* gate blocks */}
        {[
          { x: 170, sym: "σ", name: "r_t", v: st.r, on: active("reset"), show: shown.r },
          { x: 300, sym: "σ", name: "z_t", v: st.z, on: active("update"), show: shown.z },
          { x: 440, sym: "tanh", name: "h̃_t", v: st.htilde, on: active("candidate"), show: shown.c },
        ].map((g) => (
          <g key={g.name}>
            <rect x={g.x - 23} y={190} width={46} height={30} rx={5} fill={gateFill(g.on)} stroke={stroke(g.on)} strokeWidth={sw(g.on)} />
            <text x={g.x} y={210} textAnchor="middle" fontSize={13} fill="var(--text)">
              {g.sym}
            </text>
            <text x={g.x - 26} y={186} textAnchor="end" fontSize={11} fill="var(--text-muted)">
              {g.name}
            </text>
            <Bars x={g.x + 28} y={205} values={g.v} show={g.show} />
          </g>
        ))}
      </svg>

      {/* vector values */}
      <div className="mt-3 grid grid-cols-1 gap-x-6 gap-y-1 font-mono text-xs sm:grid-cols-2">
        {rows.map((r) => (
          <div key={r.name} className={`flex items-center gap-2 rounded px-1.5 py-0.5 ${r.on ? "bg-surface-2" : ""}`} style={r.on ? { outline: "1px solid var(--gold)" } : undefined}>
            <span className="w-20 shrink-0 text-muted">{r.name}</span>
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
      <div className="mt-4 text-xs text-muted">Timeline: the input, update gate z_t and hidden state h_t at each step so far (one column per step).</div>
      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${tlW} ${tlH}`} className="h-auto w-full" style={{ minWidth: Math.min(tlW, 520) }} role="img" aria-label="Timeline of update gate and hidden state across time steps">
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
                  const v = row.get(q);
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
      <div className="mt-1 font-mono text-[11px] text-muted">average update gate over steps 1..{t + 1}: {fmt(meanZ, 3)}</div>

      {/* memory probe: GRU vs vanilla RNN */}
      <div className="mt-4 rounded-lg border border-border p-3">
        <div className="text-xs font-medium">Memory test: GRU vs plain RNN on the same sequence</div>
        <div className="mb-2 text-xs text-muted">
          We add 1 to the first input x_1 and run both models again. Each bar shows ‖Δh_t‖: how much the state at step t still changes because of that first input. A model with long memory keeps its bars tall.
        </div>
        <div className="overflow-x-auto">
          <svg viewBox={`0 0 ${tlW} ${pSvgH}`} className="h-auto w-full" style={{ minWidth: Math.min(tlW, 520) }} role="img" aria-label="Memory probe: change in hidden state caused by perturbing the first input, for the GRU and a vanilla RNN">
            {pRows.map((row, r) => {
              const y0 = 16 + r * (pH + 18) + pH;
              return (
                <g key={row.label}>
                  <text x={labelW - 6} y={y0 - pH / 2} textAnchor="end" fontSize={10} fill="var(--text-muted)">
                    {row.label}
                  </text>
                  <line x1={labelW} x2={tlW - 4} y1={y0} y2={y0} stroke="var(--border)" />
                  {row.v.map((v, tau) => {
                    const vis = tau <= t;
                    const hgt = (v / pMax) * pH;
                    return (
                      <g key={tau}>
                        {vis && <rect x={labelW + tau * colW + 10} y={y0 - hgt} width={colW - 20} height={Math.max(0.5, hgt)} fill={row.color} opacity={tau === t ? 1 : 0.7} />}
                        {vis && (
                          <text x={labelW + tau * colW + colW / 2} y={y0 + 12} textAnchor="middle" fontSize={9} fill="var(--text)">
                            {fmt(v, 3)}
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
        <div className="mt-1 flex flex-wrap gap-x-6 gap-y-1 font-mono text-[11px] text-muted">
          <span>GRU: ‖Δh_{t + 1}‖ / ‖Δh_1‖ = {fmt(ratio(probe.gru), 3)}</span>
          <span>RNN: ‖Δh_{t + 1}‖ / ‖Δh_1‖ = {fmt(ratio(probe.rnn), 3)}</span>
        </div>
      </div>
    </SimShell>
  );
}
