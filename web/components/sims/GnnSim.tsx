"use client";

import { useMemo, useState, type KeyboardEvent } from "react";
import { Shuffle } from "lucide-react";
import { Controls, Narration, SimShell, Slider, fmt, signedColor, useStepper } from "./SimShell";
import { GRAPHS, gcnLimit, neighbourhoods, propagate, randomFeatures, tildeDegrees, type GnnMode } from "@/lib/sim/gnn";

const LAYERS = 64;
const W = 320;
const H = 220;
const R = 15;
const CW = 300;
const CH = 96;
const CP = { l: 34, r: 8, t: 8, b: 18 };

export default function GnnSim() {
  const [gi, setGi] = useState(0);
  const [mode, setMode] = useState<GnnMode>("gcn");
  const [aL, setAL] = useState(1);
  const [aR, setAR] = useState(1.5);
  const [seed, setSeed] = useState(0); // 0 = preset features
  const [sel, setSel] = useState(0);
  const s = useStepper(LAYERS + 1, { baseMs: 900 });

  const g = GRAPHS[gi];
  const x0 = useMemo(() => (seed === 0 ? g.x0 : randomFeatures(g, seed)), [g, seed]);
  const prop = useMemo(() => propagate(g, x0, mode, LAYERS, aL, aR, 0.2), [g, x0, mode, aL, aR]);
  const nb = useMemo(() => neighbourhoods(g), [g]);
  const dt = useMemo(() => tildeDegrees(g), [g]);
  const limit = useMemo(() => gcnLimit(g, x0), [g, x0]);

  const k = s.step;
  const h = prop.H[k];
  const maxAbs = Math.max(1e-6, ...x0.map(Math.abs));
  const node = Math.min(sel, g.n - 1);
  const Mprev = k > 0 ? prop.M[k - 1] : null; // matrix that produced h from H[k-1]
  const hPrev = k > 0 ? prop.H[k - 1] : null;
  // weights shown on the edges: the aggregation that produced the CURRENT layer (or, at layer 0, the next one)
  const Mshow = Mprev ?? prop.M[0];

  const pickGraph = (i: number) => {
    setGi(i);
    setSeed(0);
    setSel(0);
    s.reset();
  };
  const onKey = (i: number) => (e: KeyboardEvent<SVGGElement>) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      setSel(i);
    }
  };

  /* ---------- energy chart (log10 of energy relative to layer 0) ---------- */
  const e0 = Math.max(prop.energy[0], 1e-12);
  const rel = prop.energy.map((e) => Math.max(e / e0, 1e-8));
  const yMin = -8;
  const cx = (i: number) => CP.l + (i / LAYERS) * (CW - CP.l - CP.r);
  const cy = (v: number) => CP.t + (Math.log10(v) / yMin) * (CH - CP.t - CP.b);
  let path = "";
  rel.forEach((v, i) => (path += `${i ? "L" : "M"}${cx(i).toFixed(1)},${cy(v).toFixed(1)} `));

  /* ---------- the aggregation sum for the selected node ---------- */
  const terms = Mprev && hPrev ? nb[node].map((j) => ({ j, w: Mprev[node][j], hj: hPrev[j] })) : [];

  /* ---------- narration ---------- */
  const spread = Math.max(...h) - Math.min(...h);
  let title: string;
  let story: string;
  if (k === 0) {
    title = "Layer 0: the input features";
    story = `${g.blurb} Press Step to run one round of message passing. Each node swaps its value for a weighted average of its own value and its neighbours' values.`;
  } else if (k <= 2) {
    title = `Layer ${k}: ${k}-hop neighbourhoods`;
    story =
      mode === "gcn"
        ? `Each node now mixes in information from up to ${k} hop${k > 1 ? "s" : ""} away. The weights 1/√(d̃ᵢd̃ⱼ) are set by the graph's shape alone. A message from a node with many links (high degree) is turned down, and a node with many neighbours averages over more of them.`
        : `Each node now mixes in information from up to ${k} hop${k > 1 ? "s" : ""} away. GAT recomputes its weights from the current features at every layer, so the edge weights change from layer to layer. Each node's weights still add up to 1.`;
    if (g.group && k === 1) story += " Look at the noisy nodes 2 and 7: after one layer they were pulled toward their group. This smoothing is why shallow GNNs work well when neighbours tend to share labels (homophily).";
    if (g.key === "path") story += ` On the chain, node ${k} is now the farthest node with a non-zero value. A k-layer GNN can only see k hops away (its receptive field).`;
  } else if (k < 10) {
    title = `Layer ${k}: smoothing continues`;
    story = `The energy curve keeps falling, so neighbours are getting more alike. The spread of values is now ${fmt(spread, 3)} (it was ${fmt(Math.max(...x0) - Math.min(...x0), 3)}).`;
  } else {
    title = `Layer ${k}: over-smoothing`;
    story =
      mode === "gcn"
        ? `After many layers the features approach the limit c·√d̃ᵢ (the top eigenvector of Â). It depends only on each node's degree and one number, c = ⟨x, u⟩. Whatever made the nodes different is gone. Here the largest distance to that limit is ${fmt(Math.max(...h.map((v, i) => Math.abs(v - limit[i]))), 4)}.`
        : `Averaging again and again with attention weights pushes every node to almost the same value (spread ${fmt(spread, 4)}). Different weights change how fast, not where it ends up.`;
    if (g.group) {
      const signA = h.slice(0, 4).every((v) => v > 0) && h.slice(4).every((v) => v < 0);
      story += signA
        ? " The sign still splits the two groups. The single bridge edge is a bottleneck, so the second eigenvalue of Â is close to 1 and smoothing between the groups is slow. Keep stepping."
        : " The sign no longer splits the two groups. That is why most GCNs use only 2–3 layers.";
    }
  }

  return (
    <SimShell
      title="Graph Message Passing"
      subtitle="One number per node, passed along layer by layer with GCN's averaging or GAT's attention weights. W = 1 and no non-linearity, so you see only the mixing."
      controls={
        <div className="space-y-2.5">
          <Controls
            s={s}
            extra={
              <button className="sim-btn" onClick={() => { setSeed((v) => v + 1); s.reset(); }} aria-label="Draw new random node features">
                <Shuffle size={14} /> New features
              </button>
            }
          />
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-xs text-muted">graph</span>
            {GRAPHS.map((gr, i) => (
              <button key={gr.key} className={`sim-btn ${gi === i ? "primary" : ""}`} onClick={() => pickGraph(i)} aria-pressed={gi === i}>
                {gr.name}
              </button>
            ))}
            <span className="ml-2 text-xs text-muted">layer</span>
            {(["gcn", "gat"] as const).map((m) => (
              <button key={m} className={`sim-btn ${mode === m ? "primary" : ""}`} onClick={() => setMode(m)} aria-pressed={mode === m}>
                {m.toUpperCase()}
              </button>
            ))}
          </div>
          {mode === "gat" && (
            <div className="grid gap-1.5 sm:grid-cols-2">
              <Slider label="a_l (self)" value={aL} min={-3} max={3} step={0.1} onChange={setAL} fmt={(v) => fmt(v, 1)} />
              <Slider label="a_r (neighbour)" value={aR} min={-3} max={3} step={0.1} onChange={setAR} fmt={(v) => fmt(v, 1)} />
            </div>
          )}
        </div>
      }
      aside={
        <div className="space-y-3">
          <div className="rounded-md bg-surface-2 p-2 font-mono text-[11px] leading-relaxed">
            <div>
              layer {k} · node {node} · d̃ = {dt[node]} (deg {dt[node] - 1} + self-loop)
            </div>
            {k === 0 ? (
              <div className="text-muted">h{"⁽⁰⁾"} = {fmt(h[node], 3)} (input). Press Step to see how layer 1 is computed.</div>
            ) : (
              <>
                <div className="mt-1 text-muted">h{`_${node}`}⁽{k}⁾ = Σⱼ {mode === "gcn" ? "Âᵢⱼ" : "αᵢⱼ"} · hⱼ⁽{k - 1}⁾</div>
                {terms.map((t) => (
                  <div key={t.j}>
                    {t.j === node ? "self " : `n${t.j}   `}
                    {fmt(t.w, 3)} × {fmt(t.hj, 3)} = {fmt(t.w * t.hj, 3)}
                    {mode === "gat" && <span className="text-muted"> (e = {fmt(prop.scores[k - 1][node][t.j], 2)})</span>}
                  </div>
                ))}
                <div className="text-accent">
                  sum = {fmt(h[node], 3)} · weights sum to {fmt(terms.reduce((a, t) => a + t.w, 0), 3)}
                </div>
              </>
            )}
            <div className="mt-1 text-muted">
              {mode === "gcn" ? "normalised energy hᵀ(I − Â)h" : "energy Σ_edges (hᵢ − hⱼ)²"} = {fmt(prop.energy[k], 4)}
            </div>
          </div>
          <Narration title={title}>{story}</Narration>
          <p className="text-xs text-muted">Click (or tab to) a node to see how its value is mixed.</p>
        </div>
      }
    >
      <figure className="min-w-0">
        <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`${g.name} graph after ${k} ${mode.toUpperCase()} layers; node colours show feature values`}>
          {g.edges.map(([a, b], ei) => {
            const [x1, y1] = g.pos[a];
            const [x2, y2] = g.pos[b];
            const touches = a === node || b === node;
            const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
            // GAT: label α_{node←other} near the selected node's end of its incident edges
            const other = a === node ? b : a;
            const lx = mode === "gat" ? g.pos[node][0] + (g.pos[other][0] - g.pos[node][0]) * 0.55 : mx;
            const ly = mode === "gat" ? g.pos[node][1] + (g.pos[other][1] - g.pos[node][1]) * 0.55 : my;
            const label = mode === "gcn" ? fmt(Mshow[a][b], 2) : touches ? fmt(Mshow[node][other], 2) : null;
            const wgt = mode === "gcn" ? Mshow[a][b] : touches ? Mshow[node][other] : 0;
            return (
              <g key={ei}>
                <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={touches ? "var(--gold)" : "var(--border)"} strokeWidth={touches ? 1 + 5 * wgt : 1.4} strokeOpacity={touches ? 0.75 : 1} />
                {label !== null && (
                  <g>
                    <rect x={lx - 13} y={ly - 7} width={26} height={13} rx={3} fill="var(--surface)" stroke="var(--border)" strokeWidth={0.5} />
                    <text x={lx} y={ly + 3} fontSize="8.5" textAnchor="middle" fill={touches ? "var(--text)" : "var(--text-muted)"}>
                      {label}
                    </text>
                  </g>
                )}
              </g>
            );
          })}
          {g.pos.map(([x, y], i) => (
            <g key={i} role="button" tabIndex={0} aria-label={`Node ${i}, value ${fmt(h[i], 3)}`} onClick={() => setSel(i)} onKeyDown={onKey(i)} style={{ cursor: "pointer" }}>
              <circle cx={x} cy={y} r={R} fill="var(--surface)" />
              <circle cx={x} cy={y} r={R} fill={signedColor(h[i], maxAbs)} stroke={i === node ? "var(--gold)" : "var(--border)"} strokeWidth={i === node ? 3 : 1} />
              <text x={x} y={y + 3} fontSize="9" textAnchor="middle" fill="var(--text)" fontWeight={600}>
                {fmt(h[i], 2)}
              </text>
              <text x={x} y={y - R - 3} fontSize="8" textAnchor="middle" fill="var(--text-muted)">
                {i}
                {i === node && Mshow ? ` · self ${fmt(Mshow[i][i], 2)}` : ""}
              </text>
            </g>
          ))}
        </svg>
        <figcaption className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted">
          <span className="text-accent">● positive</span>
          <span className="text-clay">● negative</span>
          <span>colours fixed to the input range ±{fmt(maxAbs, 2)}</span>
          <span className="text-gold">
            {mode === "gcn" ? "edge labels: Âᵢⱼ = 1/√(d̃ᵢd̃ⱼ)" : `edge labels: α from node ${node}'s softmax`}
            {k > 0 ? ` (layer ${k - 1}→${k})` : " (layer 0→1)"}
          </span>
        </figcaption>
      </figure>

      <figure className="mt-3 min-w-0">
        <svg viewBox={`0 0 ${CW} ${CH}`} className="h-auto w-full" role="img" aria-label="Energy relative to the input, on a log scale, versus number of layers">
          {[0, -2, -4, -6, -8].map((p) => (
            <g key={p}>
              <line x1={CP.l} x2={CW - CP.r} y1={cy(10 ** p)} y2={cy(10 ** p)} stroke="var(--border)" strokeWidth={0.6} />
              <text x={CP.l - 4} y={cy(10 ** p) + 3} fontSize="8.5" textAnchor="end" fill="var(--text-muted)">
                {p === 0 ? "1" : `1e${p}`}
              </text>
            </g>
          ))}
          {[0, 16, 32, 48, 64].map((x) => (
            <text key={x} x={cx(x)} y={CH - 5} fontSize="8.5" textAnchor="middle" fill="var(--text-muted)">
              {x}
            </text>
          ))}
          <path d={path} fill="none" stroke="var(--accent)" strokeWidth={1.8} />
          <line x1={cx(k)} x2={cx(k)} y1={CP.t} y2={CH - CP.b} stroke="var(--gold)" strokeWidth={1} />
          <circle cx={cx(k)} cy={cy(rel[k])} r={3} fill="var(--gold)" />
        </svg>
        <figcaption className="text-xs text-muted">
          <span className="text-accent">{mode === "gcn" ? "normalised Dirichlet energy" : "Dirichlet energy"}</span> relative to layer 0 (log scale) vs number of layers. It measures how different neighbours still are. It falls toward 0: over-smoothing.
        </figcaption>
      </figure>
    </SimShell>
  );
}
