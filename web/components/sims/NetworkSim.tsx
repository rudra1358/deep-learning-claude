"use client";

import { useEffect, useMemo, useState } from "react";
import { Minus, Plus, Shuffle, Zap } from "lucide-react";
import { Controls, Narration, SimShell, Slider, fmt, signedColor, useStepper } from "./SimShell";
import { renderTex } from "@/components/content/Inline";
import { ACTS, applyUpdate, buildSteps, forwardBackward, initNet, numericalGrad, revealed, type Act, type LossKind, type Net } from "@/lib/sim/nn";

const W_SVG = 760;
const H_SVG = 380;
const R = 24;

type Sel = { kind: "edge"; l: number; j: number; k: number } | { kind: "node"; l: number; j: number } | null;

function Tex({ tex, display = false }: { tex: string; display?: boolean }) {
  return <span className="block overflow-x-auto" dangerouslySetInnerHTML={{ __html: renderTex(tex, display) }} />;
}

function resizeNet(net: Net, sizes: number[], acts: Act[], seed: number): Net {
  // Keep existing weights where shapes overlap so editing architecture doesn't discard the learner's edits.
  const fresh = initNet(sizes, acts, seed);
  return {
    sizes,
    acts,
    W: fresh.W.map((Wl, l) => Wl.map((row, j) => row.map((w, k) => net.W[l]?.[j]?.[k] ?? w))),
    b: fresh.b.map((bl, l) => bl.map((bv, j) => net.b[l]?.[j] ?? bv)),
  };
}

export default function NetworkSim() {
  const [seed, setSeed] = useState(7);
  const [net, setNet] = useState<Net>(() => initNet([2, 3, 1], ["sigmoid", "sigmoid"], 7));
  const [x, setX] = useState<number[]>([0.5, -1.0]);
  const [y, setY] = useState<number[]>([1]);
  const [lossKind, setLossKind] = useState<LossKind>("mse");
  const [lr, setLr] = useState(0.5);
  const [sel, setSel] = useState<Sel>(null);
  const [history, setHistory] = useState<number[]>([]);
  const [autoLoop, setAutoLoop] = useState(false);

  const steps = useMemo(() => buildSteps(net.sizes), [net.sizes]);
  const s = useStepper(steps.length, { baseMs: 1100 });
  const cur = steps[Math.min(s.step, steps.length - 1)];
  const tr = useMemo(() => forwardBackward(net, x, y, lossKind), [net, x, y, lossKind]);
  const L = net.sizes.length - 1;
  const updated = useMemo(() => applyUpdate(net, tr, lr), [net, tr, lr]);
  const lossAfter = useMemo(() => forwardBackward(updated, x, y, lossKind).loss, [updated, x, y, lossKind]);

  const commit = () => {
    setHistory((h) => [...h, tr.loss].slice(-200));
    setNet(updated);
    s.setStep(0);
  };

  // Auto-loop: when playing reaches the update step, commit and keep training.
  useEffect(() => {
    if (!autoLoop || cur.kind !== "update") return;
    const id = setTimeout(() => {
      commit();
      s.setPlaying(true);
    }, 1100 / s.speed);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoLoop, cur.kind, s.step]);

  const isFwd = (l: number, j: number) => l === 0 || revealed(steps, s.step, "forward", l, j);
  const isBwd = (l: number, j: number) => l > 0 && revealed(steps, s.step, "backward", l, j);
  const lossShown = steps.slice(0, s.step + 1).some((st) => st.kind === "loss");

  // ---- architecture editing ----
  const setArch = (sizes: number[], acts: Act[]) => {
    setNet((n) => resizeNet(n, sizes, acts, seed));
    if (sizes[0] !== x.length) setX(Array.from({ length: sizes[0] }, (_, i) => x[i] ?? 0.5));
    if (sizes[sizes.length - 1] !== y.length) setY(Array.from({ length: sizes[sizes.length - 1] }, (_, i) => y[i] ?? 1));
    setSel(null);
    s.reset();
    setHistory([]);
  };
  const changeSize = (l: number, d: number) => {
    const sizes = net.sizes.slice();
    sizes[l] = Math.max(1, Math.min(l === 0 ? 4 : l === L ? 3 : 5, sizes[l] + d));
    setArch(sizes, net.acts);
  };
  const addLayer = () => {
    if (L >= 4) return;
    setArch([...net.sizes.slice(0, L), 2, net.sizes[L]], [...net.acts.slice(0, L - 1), "relu", net.acts[L - 1]]);
  };
  const removeLayer = (l: number) => {
    if (L <= 1) return;
    setArch(net.sizes.filter((_, i) => i !== l), net.acts.filter((_, i) => i !== l - 1));
  };
  const setAct = (l: number, a: Act) => {
    const acts = net.acts.slice();
    acts[l - 1] = a;
    setNet({ ...net, acts });
  };
  const reinit = () => {
    const ns = seed + 1;
    setSeed(ns);
    setNet(initNet(net.sizes, net.acts, ns));
    setHistory([]);
    s.reset();
  };
  const trainN = (n: number) => {
    let cur = net;
    const h: number[] = [];
    for (let i = 0; i < n; i++) {
      const t = forwardBackward(cur, x, y, lossKind);
      h.push(t.loss);
      cur = applyUpdate(cur, t, lr);
    }
    setHistory((old) => [...old, ...h].slice(-200));
    setNet(cur);
    s.reset();
  };

  // ---- geometry ----
  const colX = (l: number) => 70 + (l * (W_SVG - 140)) / L;
  const rowY = (l: number, j: number) => {
    const n = net.sizes[l];
    const gap = Math.min(90, (H_SVG - 90) / Math.max(1, n));
    return H_SVG / 2 + 10 + (j - (n - 1) / 2) * gap;
  };
  const maxW = Math.max(0.5, ...net.W.flat(2).map(Math.abs));
  const activeNode = cur.kind === "forward" || cur.kind === "backward" ? { l: cur.layer, j: cur.neuron } : null;
  const pulseMs = 1100 / s.speed;

  // ---- narration for the current step ----
  const narration = (() => {
    if (cur.kind === "idle")
      return {
        title: "Ready",
        body: (
          <>
            Press <strong>Step</strong> or <strong>Play</strong>. Each step works out one neuron. Click a connection to change its weight, or a neuron to change its bias or input.
          </>
        ),
      };
    if (cur.kind === "forward") {
      const { layer: l, neuron: j } = cur;
      const terms = net.W[l - 1][j].map((w, k) => `(${fmt(w, 2)})(${fmt(tr.a[l - 1][k])})`).join(" + ");
      return {
        title: `Forward · layer ${l}, neuron ${j + 1}`,
        body: (
          <>
            <div>Multiply each output of the previous layer by its weight, add them up, then add the bias:</div>
            <Tex display tex={String.raw`z^{${l}}_{${j + 1}} = ${terms} + (${fmt(net.b[l - 1][j], 2)}) = ${fmt(tr.z[l][j])}`} />
            <div>Then pass it through the activation ({ACTS[net.acts[l - 1]].label}):</div>
            <Tex display tex={String.raw`a^{${l}}_{${j + 1}} = ${ACTS[net.acts[l - 1]].sym}\big(${fmt(tr.z[l][j])}\big) = ${fmt(tr.a[l][j])}`} />
          </>
        ),
      };
    }
    if (cur.kind === "loss") {
      const out = tr.a[L];
      const tex =
        lossKind === "mse"
          ? String.raw`L = \tfrac{1}{${out.length}}\sum_i (a_i - y_i)^2 = \tfrac{1}{${out.length}}\left[${out.map((a, i) => `(${fmt(a)}-${fmt(y[i], 2)})^2`).join("+")}\right] = ${fmt(tr.loss, 4)}`
          : String.raw`L = -\tfrac{1}{${out.length}}\sum_i \left[y_i\log a_i + (1-y_i)\log(1-a_i)\right] = ${fmt(tr.loss, 4)}`;
      return {
        title: "Loss",
        body: (
          <>
            <div>Compare the prediction with the target to see how wrong it is:</div>
            <Tex display tex={tex} />
            <div>Backpropagation starts here: how much the loss changes when each output changes:</div>
            <Tex display tex={String.raw`\frac{\partial L}{\partial a^{${L}}} = [${tr.dLda.map((v) => fmt(v)).join(",\\;")}]`} />
          </>
        ),
      };
    }
    if (cur.kind === "backward") {
      const { layer: l, neuron: j } = cur;
      const dfz = ACTS[net.acts[l - 1]].df(tr.z[l][j]);
      const upstream =
        l === L
          ? String.raw`\frac{\partial L}{\partial a^{${l}}_{${j + 1}}} = ${fmt(tr.dLda[j])}`
          : String.raw`\sum_m W^{${l + 1}}_{m,${j + 1}}\,\delta^{${l + 1}}_m = ${net.W[l]
              .map((row, m) => `(${fmt(row[j], 2)})(${fmt(tr.delta[l + 1][m])})`)
              .join("+")} = ${fmt(tr.delta[l][j] / (dfz || 1))}`;
      return {
        title: `Backward · layer ${l}, neuron ${j + 1}`,
        body: (
          <>
            <div>{l === L ? "Gradient coming in from the loss:" : "Gradient flowing back from the next layer (chain rule, added up over every path):"}</div>
            <Tex display tex={upstream} />
            <div>Multiply by the activation&apos;s slope here, f′(z) = {fmt(dfz)}:</div>
            <Tex display tex={String.raw`\delta^{${l}}_{${j + 1}} = \frac{\partial L}{\partial z^{${l}}_{${j + 1}}} = ${fmt(tr.delta[l][j])}`} />
            <div>So the gradients for this neuron&apos;s incoming weights and bias are:</div>
            <Tex
              display
              tex={String.raw`\frac{\partial L}{\partial W^{${l}}_{${j + 1},k}} = \delta^{${l}}_{${j + 1}}\, a^{${l - 1}}_k = [${tr.dW[l - 1][j].map((g) => fmt(g)).join(",\\;")}],\quad \frac{\partial L}{\partial b^{${l}}_{${j + 1}}} = ${fmt(tr.db[l - 1][j])}`}
            />
          </>
        ),
      };
    }
    return {
      title: "Gradient descent update",
      body: (
        <>
          <Tex display tex={String.raw`\theta \leftarrow \theta - \eta\,\frac{\partial L}{\partial \theta},\qquad \eta = ${lr}`} />
          <div className="max-h-40 overflow-auto font-mono text-xs">
            {net.W.map((Wl, l) =>
              Wl.map((row, j) =>
                row.map((w, k) => (
                  <div key={`${l}-${j}-${k}`}>
                    W{l + 1}[{j + 1},{k + 1}]: {fmt(w)} − {lr}×({fmt(tr.dW[l][j][k])}) = <span className="text-accent">{fmt(updated.W[l][j][k])}</span>
                  </div>
                )),
              ),
            )}
            {net.b.map((bl, l) =>
              bl.map((bv, j) => (
                <div key={`b${l}-${j}`}>
                  b{l + 1}[{j + 1}]: {fmt(bv)} − {lr}×({fmt(tr.db[l][j])}) = <span className="text-accent">{fmt(updated.b[l][j])}</span>
                </div>
              )),
            )}
          </div>
          <div className="mt-2">
            Loss before: <strong>{fmt(tr.loss, 5)}</strong> → after this update: <strong className={lossAfter < tr.loss ? "text-accent" : "text-clay"}>{fmt(lossAfter, 5)}</strong>
          </div>
          <button className="sim-btn primary mt-2" onClick={commit}>
            Apply update → next round
          </button>
        </>
      ),
    };
  })();

  // ---- inspector for selected element ----
  const inspector = (() => {
    if (!sel) return <div className="text-faint">Click a connection or neuron to see and change it.</div>;
    if (sel.kind === "edge") {
      const w = net.W[sel.l][sel.j][sel.k];
      const g = tr.dW[sel.l][sel.j][sel.k];
      const num = numericalGrad(net, x, y, lossKind, sel.l, sel.j, sel.k);
      const relErr = Math.abs(g - num) / Math.max(1e-12, Math.abs(g) + Math.abs(num));
      return (
        <div className="space-y-2">
          <div className="font-medium">
            Weight W<sup>{sel.l + 1}</sup>
            <sub>
              {sel.j + 1},{sel.k + 1}
            </sub>
          </div>
          <Slider label="value" min={-3} max={3} step={0.01} value={w} onChange={(v) => setNet({ ...net, W: net.W.map((Wl, l) => Wl.map((row, j) => row.map((ww, k) => (l === sel.l && j === sel.j && k === sel.k ? v : ww)))) })} />
          <div className="text-xs text-muted">
            Backprop gradient: <span className="font-mono text-clay">{fmt(g, 6)}</span>
            <br />
            Numerical check (nudge w both ways): <span className="font-mono">{fmt(num, 6)}</span>
            <br />
            Relative error: <span className="font-mono">{relErr.toExponential(2)}</span> {relErr < 1e-5 ? "✓ gradients match" : ""}
          </div>
        </div>
      );
    }
    if (sel.l === 0)
      return (
        <div className="space-y-2">
          <div className="font-medium">Input x{sel.j + 1}</div>
          <Slider label="value" min={-2} max={2} value={x[sel.j]} onChange={(v) => setX(x.map((xx, i) => (i === sel.j ? v : xx)))} />
        </div>
      );
    return (
      <div className="space-y-2">
        <div className="font-medium">
          Neuron {sel.j + 1} of layer {sel.l}
        </div>
        <Slider label="bias b" min={-3} max={3} value={net.b[sel.l - 1][sel.j]} onChange={(v) => setNet({ ...net, b: net.b.map((bl, l) => bl.map((bb, j) => (l === sel.l - 1 && j === sel.j ? v : bb))) })} />
        <div className="font-mono text-xs text-muted">
          z = {fmt(tr.z[sel.l][sel.j])}, a = {fmt(tr.a[sel.l][sel.j])}, δ = {fmt(tr.delta[sel.l][sel.j])}
        </div>
      </div>
    );
  })();

  const spark = (() => {
    if (history.length < 2) return null;
    const max = Math.max(...history), min = Math.min(...history);
    const pts = history.map((v, i) => `${(i / (history.length - 1)) * 260},${58 - ((v - min) / (max - min || 1)) * 52}`).join(" ");
    return (
      <svg viewBox="0 0 260 62" className="h-auto w-full" role="img" aria-label="Loss history">
        <polyline points={pts} fill="none" stroke="var(--accent)" strokeWidth={1.6} />
      </svg>
    );
  })();

  return (
    <SimShell
      title="Neural Network Visualizer"
      subtitle="Build a network, then watch it really run: forward pass, backpropagation and gradient descent."
      controls={
        <Controls
          s={s}
          extra={
            <label className="flex items-center gap-1.5 text-xs text-muted">
              <input type="checkbox" checked={autoLoop} onChange={(e) => setAutoLoop(e.target.checked)} /> keep training
            </label>
          }
        />
      }
      aside={
        <div className="space-y-4">
          <Narration title={narration.title}>{narration.body}</Narration>
          <div>
            <div className="mb-1 text-xs font-medium uppercase tracking-wider text-faint">Inspector</div>
            {inspector}
          </div>
          <div>
            <div className="mb-1 flex justify-between text-xs font-medium uppercase tracking-wider text-faint">
              <span>Loss history</span>
              {history.length > 0 && <span className="font-mono normal-case">{history.length} iters · last {fmt(history.at(-1)!, 4)}</span>}
            </div>
            {spark ?? <div className="text-xs text-faint">Apply updates or press “Train ×25” to watch the loss drop.</div>}
          </div>
        </div>
      }
    >
      {/* Architecture & data controls */}
      <div className="mb-3 flex flex-wrap items-end gap-x-5 gap-y-3 text-xs">
        {net.sizes.map((n, l) => (
          <div key={l} className="flex flex-col gap-1">
            <span className="text-muted">{l === 0 ? "Inputs" : l === L ? "Output" : `Hidden ${l}`}</span>
            <div className="flex items-center gap-1">
              <button className="sim-btn !px-1.5 !py-1" onClick={() => changeSize(l, -1)} aria-label={`Remove neuron from layer ${l}`}>
                <Minus size={12} />
              </button>
              <span className="w-4 text-center font-mono">{n}</span>
              <button className="sim-btn !px-1.5 !py-1" onClick={() => changeSize(l, 1)} aria-label={`Add neuron to layer ${l}`}>
                <Plus size={12} />
              </button>
              {l > 0 && (
                <select className="field !py-1 !text-xs" value={net.acts[l - 1]} onChange={(e) => setAct(l, e.target.value as Act)} aria-label={`Activation of layer ${l}`}>
                  {(Object.keys(ACTS) as Act[]).map((a) => (
                    <option key={a} value={a}>
                      {ACTS[a].label}
                    </option>
                  ))}
                </select>
              )}
              {l > 0 && l < L && (
                <button className="text-faint hover:text-clay" onClick={() => removeLayer(l)} aria-label={`Remove hidden layer ${l}`}>
                  ×
                </button>
              )}
            </div>
          </div>
        ))}
        <button className="sim-btn" onClick={addLayer} disabled={L >= 4}>
          <Plus size={13} /> Layer
        </button>
        <label className="flex flex-col gap-1">
          <span className="text-muted">Loss</span>
          <select className="field !py-1 !text-xs" value={lossKind} onChange={(e) => setLossKind(e.target.value as LossKind)}>
            <option value="mse">MSE</option>
            <option value="bce">Binary cross-entropy</option>
          </select>
        </label>
      </div>
      <div className="mb-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs">
        <span className="text-muted">x =</span>
        {x.map((v, i) => (
          <input key={i} className="field w-16 !py-1 font-mono" type="number" step={0.1} value={v} onChange={(e) => setX(x.map((xx, k) => (k === i ? +e.target.value : xx)))} aria-label={`Input ${i + 1}`} />
        ))}
        <span className="text-muted">target y =</span>
        {y.map((v, i) => (
          <input key={i} className="field w-16 !py-1 font-mono" type="number" step={0.1} min={lossKind === "bce" ? 0 : undefined} max={lossKind === "bce" ? 1 : undefined} value={v} onChange={(e) => setY(y.map((yy, k) => (k === i ? +e.target.value : yy)))} aria-label={`Target ${i + 1}`} />
        ))}
        <div className="w-56">
          <Slider label="learning rate η" min={0.01} max={3} step={0.01} value={lr} onChange={setLr} />
        </div>
        <button className="sim-btn" onClick={reinit}>
          <Shuffle size={13} /> Re-initialise
        </button>
        <button className="sim-btn" onClick={() => trainN(25)}>
          <Zap size={13} /> Train ×25
        </button>
      </div>
      {lossKind === "bce" && net.acts[L - 1] !== "sigmoid" && <div className="mb-2 rounded bg-clay-soft px-3 py-1.5 text-xs">Binary cross-entropy needs outputs between 0 and 1. Set the output activation to Sigmoid.</div>}

      <svg viewBox={`0 0 ${W_SVG} ${H_SVG}`} className="h-auto w-full select-none" role="img" aria-label="Neural network diagram with live values">
        {/* edges */}
        {net.W.map((Wl, l) =>
          Wl.map((row, j) =>
            row.map((w, k) => {
              const x1 = colX(l) + R, y1 = rowY(l, k), x2 = colX(l + 1) - R, y2 = rowY(l + 1, j);
              const fwdActive = cur.kind === "forward" && cur.layer === l + 1 && cur.neuron === j;
              const bwdActive = cur.kind === "backward" && cur.layer === l + 1 && cur.neuron === j;
              const isSel = sel?.kind === "edge" && sel.l === l && sel.j === j && sel.k === k;
              const stroke = bwdActive ? "var(--clay)" : fwdActive ? "var(--gold)" : signedColor(w, maxW);
              const mx = x1 + (x2 - x1) * 0.62, my = y1 + (y2 - y1) * 0.62;
              const showLabel = fwdActive || bwdActive || isSel;
              return (
                <g key={`${l}-${j}-${k}`} onClick={() => setSel({ kind: "edge", l, j, k })} className="cursor-pointer">
                  <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="transparent" strokeWidth={12} />
                  <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={stroke} strokeWidth={(isSel ? 2 : 0) + 1 + (2.5 * Math.abs(w)) / maxW} strokeDasharray={w < 0 && !fwdActive && !bwdActive ? "5 3" : undefined} />
                  {(fwdActive || bwdActive) && (
                    <circle r={4.5} fill={bwdActive ? "var(--clay)" : "var(--gold)"} key={`p${s.step}`}>
                      <animateMotion dur={`${pulseMs * 0.8}ms`} fill="freeze" path={bwdActive ? `M${x2},${y2} L${x1},${y1}` : `M${x1},${y1} L${x2},${y2}`} />
                    </circle>
                  )}
                  {showLabel && (
                    <g>
                      <rect x={mx - 30} y={my - 10} width={60} height={18} rx={4} fill="var(--surface)" stroke="var(--border)" />
                      <text x={mx} y={my + 3} textAnchor="middle" fontSize={10.5} fontFamily="var(--font-mono)" fill={bwdActive ? "var(--clay)" : "var(--text)"}>
                        {bwdActive ? `∂${fmt(tr.dW[l][j][k], 3)}` : fwdActive ? `${fmt(w, 2)}×${fmt(tr.a[l][k], 2)}` : `w=${fmt(w, 2)}`}
                      </text>
                    </g>
                  )}
                </g>
              );
            }),
          ),
        )}
        {/* layer labels */}
        {net.sizes.map((_, l) => (
          <text key={l} x={colX(l)} y={22} textAnchor="middle" fontSize={11.5} fill="var(--text-muted)">
            {l === 0 ? "Input" : l === L ? `Output · ${ACTS[net.acts[l - 1]].label}` : `Hidden ${l} · ${ACTS[net.acts[l - 1]].label}`}
          </text>
        ))}
        {/* neurons */}
        {net.sizes.map((n, l) =>
          Array.from({ length: n }, (_, j) => {
            const cx = colX(l), cy = rowY(l, j);
            const shown = isFwd(l, j);
            const back = isBwd(l, j);
            const active = activeNode?.l === l && activeNode.j === j;
            const isSel = sel?.kind === "node" && sel.l === l && sel.j === j;
            return (
              <g key={`${l}-${j}`} onClick={() => setSel({ kind: "node", l, j })} className="cursor-pointer">
                <circle cx={cx} cy={cy} r={R} fill={shown ? signedColor(tr.a[l][j], 1.5) : "var(--surface)"} stroke={active ? (cur.kind === "backward" ? "var(--clay)" : "var(--gold)") : isSel ? "var(--accent)" : "var(--border-strong)"} strokeWidth={active || isSel ? 2.5 : 1.2} />
                <circle cx={cx} cy={cy} r={R - 3} fill="var(--surface)" opacity={shown ? 0.55 : 1} />
                <text x={cx} y={cy + 4} textAnchor="middle" fontSize={11.5} fontFamily="var(--font-mono)" fill="var(--text)">
                  {shown ? fmt(tr.a[l][j], 2) : l === 0 ? fmt(x[j], 2) : "?"}
                </text>
                {l === 0 && (
                  <text x={cx - R - 6} y={cy + 4} textAnchor="end" fontSize={11} fill="var(--text-muted)">
                    x{j + 1}
                  </text>
                )}
                {l > 0 && shown && (
                  <text x={cx} y={cy - R - 5} textAnchor="middle" fontSize={9.5} fontFamily="var(--font-mono)" fill="var(--text-faint)">
                    z={fmt(tr.z[l][j], 2)}
                  </text>
                )}
                {back && (
                  <text x={cx} y={cy + R + 14} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--clay)">
                    δ={fmt(tr.delta[l][j], 3)}
                  </text>
                )}
                {l === L && lossShown && (
                  <text x={cx + R + 8} y={cy + 4} fontSize={10.5} fill="var(--text-muted)">
                    y={fmt(y[j], 2)}
                  </text>
                )}
              </g>
            );
          }),
        )}
        {lossShown && (
          <text x={W_SVG - 16} y={H_SVG - 14} textAnchor="end" fontSize={12.5} fontFamily="var(--font-mono)" fill="var(--text)">
            L = {fmt(tr.loss, 5)}
          </text>
        )}
        <g fontSize={10} fill="var(--text-faint)">
          <text x={14} y={H_SVG - 28}>solid = positive weight, dashed = negative; thicker = bigger |w|</text>
          <text x={14} y={H_SVG - 13}>
            <tspan fill="var(--gold)">●</tspan> forward signal&nbsp;&nbsp;<tspan fill="var(--clay)">●</tspan> backward gradient
          </text>
        </g>
      </svg>
    </SimShell>
  );
}
