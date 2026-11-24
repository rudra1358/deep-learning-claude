"use client";

import { useMemo, useState, type ReactNode } from "react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import {
  CHAIN_STEPS,
  EDGES,
  backward,
  finiteDiff,
  forward,
  type ChainParams,
  type EdgeId,
  type NodeId,
} from "@/lib/sim/chainRule";

/* ---------------- Layout ---------------- */

type NodeDef = { x: number; y: number; leaf: boolean; label: string; op?: string; expr: string };

const NODES: Record<NodeId, NodeDef> = {
  w: { x: 52, y: 70, leaf: true, label: "w", expr: "weight (input)" },
  x: { x: 52, y: 200, leaf: true, label: "x", expr: "feature (input)" },
  u: { x: 180, y: 135, leaf: false, label: "u", op: "×", expr: "u = w·x" },
  b: { x: 250, y: 250, leaf: true, label: "b", expr: "bias (input)" },
  z: { x: 315, y: 135, leaf: false, label: "z", op: "+", expr: "z = u + b" },
  a: { x: 450, y: 135, leaf: false, label: "a", op: "σ", expr: "a = σ(z)" },
  y: { x: 520, y: 250, leaf: true, label: "y", expr: "target (input)" },
  d: { x: 585, y: 135, leaf: false, label: "d", op: "−", expr: "d = a − y" },
  L: { x: 710, y: 135, leaf: false, label: "L", op: "( )²", expr: "L = d²" },
};

const ORDER: NodeId[] = ["x", "w", "b", "y", "u", "z", "a", "d", "L"];
const R = 24; // radius of op nodes
const LEAF_W = 70;
const LEAF_H = 32;

/** Short edge label: the local derivative (child w.r.t. parent) at the current inputs. */
function localLabel(e: EdgeId, v: number): string {
  switch (e) {
    case "w-u":
      return `x=${fmt(v, 2)}`;
    case "x-u":
      return `w=${fmt(v, 2)}`;
    case "z-a":
      return `σ′=${fmt(v, 3)}`;
    case "d-L":
      return `2d=${fmt(v, 3)}`;
    default:
      return v < 0 ? "−1" : "1";
  }
}

/** Shorten a segment so it starts/ends at the node boundaries. */
function endpoints(from: NodeDef, to: NodeDef) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  const pad = (n: NodeDef, dirx: number, diry: number) => {
    if (!n.leaf) return R + 2;
    // distance from rectangle centre to its boundary along the direction
    const tx = Math.abs(dirx) > 1e-9 ? LEAF_W / 2 / Math.abs(dirx) : Infinity;
    const ty = Math.abs(diry) > 1e-9 ? LEAF_H / 2 / Math.abs(diry) : Infinity;
    return Math.min(tx, ty) + 2;
  };
  const p0 = pad(from, ux, uy);
  const p1 = pad(to, ux, uy) + 4;
  return { x1: from.x + ux * p0, y1: from.y + uy * p0, x2: to.x - ux * p1, y2: to.y - uy * p1 };
}

export default function ChainRuleSim() {
  const [p, setP] = useState<ChainParams>({ x: 1.5, w: 0.8, b: 0.1, y: 1 });
  const s = useStepper(CHAIN_STEPS.length, { baseMs: 1600 });
  const cur = CHAIN_STEPS[Math.min(s.step, CHAIN_STEPS.length - 1)];

  const f = useMemo(() => forward(p), [p]);
  const g = useMemo(() => backward(p, f), [p, f]);
  const fdW = useMemo(() => finiteDiff(p, "w"), [p]);
  const fdB = useMemo(() => finiteDiff(p, "b"), [p]);

  const value: Record<NodeId, number> = { x: p.x, w: p.w, b: p.b, y: p.y, u: f.u, z: f.z, a: f.a, d: f.d, L: f.L };
  const showVal = (n: NodeId) => cur.values.includes(n);
  const showGrad = (n: NodeId) => cur.grads.includes(n);
  const isActive = (n: NodeId) => cur.active.includes(n);
  const edgeUsed = (e: EdgeId) => {
    // an edge's local derivative is "used" once the gradient has crossed it to its parent
    const edge = EDGES.find((x) => x.id === e)!;
    return cur.grads.includes(edge.from);
  };
  const edgeActive = (e: EdgeId) => cur.edges.includes(e);

  const set = (k: keyof ChainParams) => (v: number) => setP((q) => ({ ...q, [k]: v }));
  const n4 = (v: number) => fmt(v, 4);

  /* ---------------- Narration ---------------- */
  const idx = Math.min(s.step, CHAIN_STEPS.length - 1);
  let title = "";
  let body: ReactNode = null;
  const gr = g.grad;
  switch (idx) {
    case 0:
      title = "The expression as a graph";
      body = (
        <>
          L = (σ(w·x + b) − y)² is broken into five primitive operations, each a node. Inputs: x = {n4(p.x)}, w = {n4(p.w)}, b = {n4(p.b)}, y ={" "}
          {n4(p.y)}. First the <b>forward pass</b> computes every node value from left to right.
        </>
      );
      break;
    case 1:
      title = "Forward 1/5: multiply";
      body = <>u = w·x = {n4(p.w)} × {n4(p.x)} = <b>{n4(f.u)}</b>.</>;
      break;
    case 2:
      title = "Forward 2/5: add bias";
      body = <>z = u + b = {n4(f.u)} + {n4(p.b)} = <b>{n4(f.z)}</b>.</>;
      break;
    case 3:
      title = "Forward 3/5: sigmoid";
      body = <>a = σ(z) = 1 / (1 + e^(−z)) = 1 / (1 + e^({n4(-f.z)})) = <b>{n4(f.a)}</b>.</>;
      break;
    case 4:
      title = "Forward 4/5: subtract target";
      body = <>d = a − y = {n4(f.a)} − {n4(p.y)} = <b>{n4(f.d)}</b>.</>;
      break;
    case 5:
      title = "Forward 5/5: square";
      body = (
        <>
          L = d² = ({n4(f.d)})² = <b>{n4(f.L)}</b>. The forward pass is complete and every intermediate value is stored — the backward pass needs
          them.
        </>
      );
      break;
    case 6:
      title = "Backward: seed the gradient";
      body = <>Start at the output with ∂L/∂L = <b>1</b>. Every other gradient is this 1 multiplied by local derivatives along the path back.</>;
      break;
    case 7:
      title = "Backward: through the square";
      body = (
        <>
          Local derivative of d² is 2d = {n4(g.local["d-L"])}. Upstream × local: ∂L/∂d = 1 × {n4(g.local["d-L"])} = <b>{n4(gr.d)}</b>.
        </>
      );
      break;
    case 8:
      title = "Backward: through the subtraction";
      body = (
        <>
          ∂d/∂a = 1 and ∂d/∂y = −1, so subtraction passes the gradient through (with a sign flip for y): ∂L/∂a = <b>{n4(gr.a)}</b>, ∂L/∂y ={" "}
          <b>{n4(gr.y)}</b>.
        </>
      );
      break;
    case 9:
      title = "Backward: through the sigmoid";
      body = (
        <>
          σ′(z) = a(1 − a) = {n4(f.a)} × {n4(1 - f.a)} = {n4(g.local["z-a"])}, using the stored forward value a. ∂L/∂z = {n4(gr.a)} ×{" "}
          {n4(g.local["z-a"])} = <b>{n4(gr.z)}</b>.
        </>
      );
      break;
    case 10:
      title = "Backward: through the addition";
      body = (
        <>
          Both local derivatives of u + b are 1, so addition copies the gradient to each input: ∂L/∂u = ∂L/∂b = <b>{n4(gr.z)}</b>.
        </>
      );
      break;
    case 11:
      title = "Backward: through the multiplication";
      body = (
        <>
          For u = w·x the local derivatives swap the inputs: ∂u/∂w = x, ∂u/∂x = w. ∂L/∂w = {n4(gr.u)} × {n4(p.x)} = <b>{n4(gr.w)}</b>; ∂L/∂x ={" "}
          {n4(gr.u)} × {n4(p.w)} = <b>{n4(gr.x)}</b>.
        </>
      );
      break;
    default:
      title = "Check against finite differences";
      body = (
        <>
          Nudge w by ±h (h = 10⁻⁴) and recompute L: (L(w+h) − L(w−h)) / 2h = <b>{n4(fdW)}</b>, matching the chain-rule value {n4(gr.w)}. One backward
          pass gave every gradient at once; finite differences need two forward passes <i>per parameter</i>.
        </>
      );
  }

  return (
    <SimShell
      title="Chain Rule on a Computational Graph"
      subtitle="L = (σ(w·x + b) − y)² — values flow forward, gradients flow backward."
      controls={<Controls s={s} />}
      aside={
        <>
          <Narration title={title}>{body}</Narration>
          <div className="mt-3 rounded-lg border border-border p-3 text-xs">
            <div className="mb-1.5 uppercase tracking-wider text-faint">Gradient check</div>
            {cur.phase === "check" ? (
              <table className="w-full font-mono">
                <thead>
                  <tr className="text-faint">
                    <th className="text-left font-normal" />
                    <th className="text-right font-normal">chain rule</th>
                    <th className="text-right font-normal">finite diff</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>∂L/∂w</td>
                    <td className="text-right text-clay">{fmt(gr.w, 6)}</td>
                    <td className="text-right">{fmt(fdW, 6)}</td>
                  </tr>
                  <tr>
                    <td>∂L/∂b</td>
                    <td className="text-right text-clay">{fmt(gr.b, 6)}</td>
                    <td className="text-right">{fmt(fdB, 6)}</td>
                  </tr>
                  <tr className="text-muted">
                    <td>|Δ| for w</td>
                    <td colSpan={2} className="text-right">
                      {Math.abs(gr.w - fdW).toExponential(1)}
                    </td>
                  </tr>
                </tbody>
              </table>
            ) : (
              <span className="text-muted">Shown after the backward pass reaches w and b.</span>
            )}
          </div>
        </>
      }
    >
      <svg viewBox="0 0 760 300" className="h-auto w-full" role="img" aria-label="Computational graph of L equals sigmoid of w x plus b, minus y, squared, with forward values and backward gradients">
        <defs>
          <marker id="cr-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--text-faint)" />
          </marker>
          <marker id="cr-arrow-clay" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--clay)" />
          </marker>
        </defs>

        {/* phase labels */}
        <text x={12} y={20} fontSize={12} fill={cur.phase === "forward" ? "var(--accent)" : "var(--text-faint)"}>
          forward: values →
        </text>
        <text x={748} y={20} fontSize={12} textAnchor="end" fill={cur.phase === "backward" || cur.phase === "check" ? "var(--clay)" : "var(--text-faint)"}>
          ← backward: gradients ∂L/∂·
        </text>

        {/* edges */}
        {EDGES.map((e) => {
          const a = NODES[e.from];
          const b = NODES[e.to];
          const { x1, y1, x2, y2 } = endpoints(a, b);
          const active = edgeActive(e.id);
          const used = edgeUsed(e.id);
          const mx = (x1 + x2) / 2;
          const my = (y1 + y2) / 2;
          const horizontal = Math.abs(y2 - y1) < 1;
          const lx = horizontal ? mx : mx + (e.from === "w" ? -8 : e.from === "x" ? -8 : 10);
          const ly = horizontal ? my - 7 : my + (e.from === "w" ? -6 : 4);
          const anchor = horizontal ? "middle" : e.from === "w" || e.from === "x" ? "end" : "start";
          return (
            <g key={e.id}>
              {active ? (
                // backward arrow: drawn from child to parent
                <line x1={x2} y1={y2} x2={x1} y2={y1} stroke="var(--clay)" strokeWidth={3} markerEnd="url(#cr-arrow-clay)" />
              ) : (
                <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--border-strong)" strokeWidth={1.5} markerEnd="url(#cr-arrow)" />
              )}
              {used && (
                <text
                  x={lx}
                  y={ly}
                  fontSize={11}
                  textAnchor={anchor}
                  fill={active ? "var(--clay)" : "var(--text-muted)"}
                  fontWeight={active ? 600 : 400}
                  fontFamily="ui-monospace, monospace"
                >
                  {localLabel(e.id, g.local[e.id])}
                </text>
              )}
            </g>
          );
        })}

        {/* nodes */}
        {ORDER.map((id) => {
          const n = NODES[id];
          const active = isActive(id);
          const stroke = active ? "var(--gold)" : "var(--border-strong)";
          const sw = active ? 2.5 : 1.5;
          const vis = showVal(id);
          return (
            <g key={id}>
              {n.leaf ? (
                <>
                  <rect x={n.x - LEAF_W / 2} y={n.y - LEAF_H / 2} width={LEAF_W} height={LEAF_H} rx={7} fill="var(--surface)" stroke={stroke} strokeWidth={sw} />
                  <text x={n.x} y={n.y + 4.5} fontSize={13} textAnchor="middle" fill="var(--accent)" fontFamily="ui-monospace, monospace">
                    {n.label}={fmt(value[id], 2)}
                  </text>
                </>
              ) : (
                <>
                  <circle cx={n.x} cy={n.y} r={R} fill="var(--surface)" stroke={stroke} strokeWidth={sw} />
                  <text x={n.x} y={n.y + 5} fontSize={n.op && n.op.length > 2 ? 13 : 17} textAnchor="middle" fill="var(--text)">
                    {n.op}
                  </text>
                  <text x={n.x} y={n.y + R + 15} fontSize={11} textAnchor="middle" fill="var(--text-muted)">
                    {n.label}
                  </text>
                  {vis && (
                    <text x={n.x} y={n.y + R + 30} fontSize={12.5} textAnchor="middle" fill="var(--accent)" fontFamily="ui-monospace, monospace">
                      {fmt(value[id], 4)}
                    </text>
                  )}
                </>
              )}
              {showGrad(id) && (
                <text
                  x={n.x}
                  y={n.leaf ? (id === "x" || id === "b" || id === "y" ? n.y + LEAF_H / 2 + 15 : n.y - LEAF_H / 2 - 7) : n.y - R - 9}
                  fontSize={12.5}
                  textAnchor="middle"
                  fill="var(--clay)"
                  fontWeight={active ? 700 : 500}
                  fontFamily="ui-monospace, monospace"
                >
                  {fmt(g.grad[id], 4)}
                </text>
              )}
            </g>
          );
        })}
      </svg>

      <div className="mt-2 grid gap-x-6 gap-y-1.5 sm:grid-cols-2">
        <Slider label="x (feature)" value={p.x} min={-3} max={3} step={0.1} onChange={set("x")} />
        <Slider label="w (weight)" value={p.w} min={-3} max={3} step={0.1} onChange={set("w")} />
        <Slider label="b (bias)" value={p.b} min={-3} max={3} step={0.1} onChange={set("b")} />
        <Slider label="y (target)" value={p.y} min={0} max={1} step={0.05} onChange={set("y")} />
      </div>

      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[320px] text-xs">
          <thead>
            <tr className="border-b border-border text-left text-faint">
              <th className="py-1 font-normal">node</th>
              <th className="py-1 font-normal">definition</th>
              <th className="py-1 text-right font-normal">value</th>
              <th className="py-1 text-right font-normal">∂L/∂node</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {(["w", "x", "b", "u", "z", "a", "y", "d", "L"] as NodeId[]).map((id) => (
              <tr key={id} className="border-b border-border/60" style={{ background: isActive(id) ? "var(--gold-soft)" : undefined }}>
                <td className="py-1">{id}</td>
                <td className="py-1 font-sans text-muted">{NODES[id].expr}</td>
                <td className="py-1 text-right text-accent">{showVal(id) ? fmt(value[id], 4) : "—"}</td>
                <td className="py-1 text-right text-clay">{showGrad(id) ? fmt(g.grad[id], 4) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs text-muted">
        <span className="text-accent">Accent</span> numbers are forward values; <span className="text-clay">clay</span> numbers are gradients ∂L/∂(node).
        Gold outlines mark the node(s) computed in the current step. Edge labels show the local derivative used when the gradient crosses that edge.
      </p>
    </SimShell>
  );
}
