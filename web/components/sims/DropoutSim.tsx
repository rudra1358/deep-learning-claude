"use client";

import { useMemo, useState } from "react";
import { Shuffle } from "lucide-react";
import { Controls, Narration, SimShell, Slider, fmt, signedColor, useStepper } from "./SimShell";
import {
  MAX_MASKS,
  N_HID,
  N_IN,
  evalScale,
  exactExpectation,
  forward,
  makeNet,
  makeUniforms,
  maskFrom,
  preActVariance,
  sampleSeries,
  trainScale,
  type OutAct,
  type Scheme,
} from "@/lib/sim/dropout";

/* network diagram geometry (viewBox units) */
const NW = 520;
const NH = 300;
const XI = 34;
const XH = 250;
const XO = 476;
const R = 10;
const yIn = (i: number) => 22 + (i * (NH - 44)) / (N_IN - 1);
const yHid = (j: number) => 16 + (j * (NH - 32)) / (N_HID - 1);
const YO = NH / 2;

/* output plot geometry */
const PW = 520;
const PH = 190;
const PP = { l: 40, r: 10, t: 10, b: 22 };

export default function DropoutSim() {
  const net = useMemo(() => makeNet(), []);
  const [p, setP] = useState(0.5);
  const [scheme, setScheme] = useState<Scheme>("inverted");
  const [out, setOut] = useState<OutAct>("linear");
  const [mode, setMode] = useState<"train" | "eval">("train");
  const [seed, setSeed] = useState(1);

  const s = useStepper(MAX_MASKS + 1, { baseMs: 700 });
  const k = s.step; // number of masks drawn so far
  const U = useMemo(() => makeUniforms(seed), [seed]);

  const full = useMemo(() => sampleSeries(net, U, MAX_MASKS, p, scheme, out), [net, U, p, scheme, out]);
  const evalPass = useMemo(() => forward(net, null, p, scheme, out), [net, p, scheme, out]);
  const exact = useMemo(() => exactExpectation(net, p, scheme, out), [net, p, scheme, out]);
  // what you would get with classic dropout if you forgot the test-time (1 − p) factor
  const naive = useMemo(() => forward(net, null, 0, "inverted", out).y, [net, out]);
  const sd = Math.sqrt(preActVariance(net, p, scheme));

  const mask = k > 0 ? maskFrom(U[k - 1], p) : null;
  const showTrain = mode === "train" && mask !== null;
  const pass = showTrain ? forward(net, mask, p, scheme, out) : evalPass;
  const runMean = k > 0 ? full.mean[k - 1] : NaN;
  const kept = mask ? mask.reduce((a, b) => a + b, 0) : N_HID;

  const maxAbsW1 = Math.max(...net.W1.flat().map(Math.abs));
  const maxContrib = Math.max(0.05, ...pass.contrib.map(Math.abs));

  /* ---- plot scales ---- */
  const allY = [...full.ys, evalPass.y, exact, ...(scheme === "classic" ? [naive] : [])];
  let yLo = Math.min(...allY);
  let yHi = Math.max(...allY);
  if (yHi - yLo < 1e-6) {
    yLo -= 0.5;
    yHi += 0.5;
  }
  const pad = 0.06 * (yHi - yLo);
  yLo -= pad;
  yHi += pad;
  const px = (j: number) => PP.l + (j / MAX_MASKS) * (PW - PP.l - PP.r);
  const py = (v: number) => PP.t + ((yHi - v) / (yHi - yLo)) * (PH - PP.t - PP.b);
  const meanPath = full.mean
    .slice(0, k)
    .map((m, j) => `${j ? "L" : "M"}${px(j + 1).toFixed(1)},${py(m).toFixed(1)}`)
    .join(" ");
  const ticks = [yLo + pad, (yLo + yHi) / 2, yHi - pad];

  const nonlinear = out === "sigmoid";
  const sT = trainScale(p, scheme);
  const sE = evalScale(p, scheme);

  /* ---- narration ---- */
  let title: string;
  let body: string;
  if (mode === "eval") {
    title = "Eval mode: no units are dropped";
    body =
      scheme === "inverted"
        ? `With inverted dropout, nothing changes at test time. Every unit passes h through as is (scale 1), giving ${out === "linear" ? "y" : "z"} = ${fmt(evalPass.z)}. The 1/(1−p) = ${fmt(1 / (1 - p), 2)} boost was already applied during training.`
        : `With classic dropout, training did not rescale the surviving units. So at test time every activation (or, the same thing, every outgoing weight) must be multiplied by 1 − p = ${fmt(1 - p, 2)}: z = ${fmt(evalPass.z)}. Forget this and you get ${fmt(forward(net, null, 0, "inverted", "linear").z)}, which is always off in the same direction.`;
  } else if (k === 0) {
    title = "Press Step to draw a dropout mask";
    body = `Each Step draws a new random mask (a coin flip per unit): each hidden unit stays with probability 1 − p = ${fmt(1 - p, 2)}. ${scheme === "inverted" ? `Units that stay are multiplied by 1/(1 − p) = ${fmt(sT, 3)}.` : "Units that stay are not rescaled (classic dropout)."}`;
  } else {
    title = `Mask ${k}: ${kept} of ${N_HID} units kept`;
    body = `This smaller network outputs ${out === "linear" ? "y" : "y = σ(z)"} = ${fmt(pass.y)}. The running mean over ${k} masks is ${fmt(runMean)}. The eval-mode output is ${fmt(evalPass.y)}${
      nonlinear ? `, and the exact average over all 1024 masks is ${fmt(exact)}` : ""
    }. ${
      nonlinear
        ? `Because σ is curved, E[σ(z)] ≠ σ(E[z]). So the eval network only comes close to the average over all masks (gap ${fmt(exact - evalPass.y)}).`
        : `Because z is a straight-line (linear) sum of the activations, E[z] equals the eval output exactly. The running mean wobbles around it with per-sample std ${fmt(sd)}, and its error shrinks like 1/√k.`
    }`;
  }

  const toggle = (label: string, on: boolean, onClick: () => void) => (
    <button className={`sim-btn ${on ? "primary" : ""}`} onClick={onClick} aria-pressed={on}>
      {label}
    </button>
  );

  return (
    <SimShell
      title="Dropout"
      subtitle="8 inputs → 10 ReLU hidden units (dropout here) → 1 output. Each Step draws a new mask. The plot compares these random smaller networks with the fixed eval-mode output."
      controls={
        <div className="space-y-2">
          <Controls
            s={s}
            extra={
              <button
                className="sim-btn"
                onClick={() => {
                  setSeed((v) => v + 1);
                  s.reset();
                }}
                aria-label="Draw a new mask sequence"
              >
                <Shuffle size={14} /> New masks
              </button>
            }
          />
          <div className="flex flex-wrap items-center gap-1.5 text-xs">
            <span className="text-muted">Mode</span>
            {toggle("Train", mode === "train", () => setMode("train"))}
            {toggle("Eval", mode === "eval", () => setMode("eval"))}
            <span className="ml-2 text-muted">Scaling</span>
            {toggle("Inverted", scheme === "inverted", () => setScheme("inverted"))}
            {toggle("Classic", scheme === "classic", () => setScheme("classic"))}
            <span className="ml-2 text-muted">Output</span>
            {toggle("Linear", out === "linear", () => setOut("linear"))}
            {toggle("Sigmoid", out === "sigmoid", () => setOut("sigmoid"))}
          </div>
        </div>
      }
      aside={
        <div className="space-y-3">
          <Slider label="drop prob p" value={p} min={0} max={0.9} step={0.05} onChange={setP} fmt={(v) => v.toFixed(2)} />
          <div className="grid grid-cols-2 gap-2 text-center text-xs">
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">train scale</div>
              <div className="font-mono text-sm">×{fmt(sT, 3)}</div>
            </div>
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">eval scale</div>
              <div className="font-mono text-sm">×{fmt(sE, 3)}</div>
            </div>
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">{showTrain ? `mask ${k} output` : "eval output"}</div>
              <div className="font-mono text-sm text-gold">{fmt(pass.y)}</div>
            </div>
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">running mean ({k})</div>
              <div className="font-mono text-sm text-accent">{k ? fmt(runMean) : "—"}</div>
            </div>
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">eval-mode output</div>
              <div className="font-mono text-sm">{fmt(evalPass.y)}</div>
            </div>
            <div className="rounded-md bg-surface-2 p-1.5">
              <div className="text-faint">exact E over 1024 masks</div>
              <div className="font-mono text-sm">{fmt(exact)}</div>
            </div>
          </div>
          <p className="text-xs text-faint">
            Std of z across masks: √(p(1−p)·s²·Σ w₂ᵢ²hᵢ²) = {fmt(sd)}{k > 1 ? `; std error of the running mean ≈ ${fmt(sd / Math.sqrt(k))}` : ""}.
          </p>
          <Narration title={title}>{body}</Narration>
        </div>
      }
    >
      <svg viewBox={`0 0 ${NW} ${NH}`} className="h-auto w-full" role="img" aria-label={`Network diagram; ${showTrain ? `${N_HID - kept} hidden units dropped` : "all hidden units active"}`}>
        {/* input → hidden edges */}
        {net.W1.map((row, j) =>
          row.map((w, i) => {
            const dropped = showTrain && mask![j] === 0;
            return (
              <line
                key={`e${i}-${j}`}
                x1={XI + R}
                y1={yIn(i)}
                x2={XH - R}
                y2={yHid(j)}
                stroke={dropped ? "var(--border)" : signedColor(w, maxAbsW1)}
                strokeWidth={0.8}
                strokeDasharray={dropped ? "3 3" : undefined}
                opacity={dropped ? 0.6 : 0.55}
              />
            );
          }),
        )}
        {/* hidden → output edges, width ∝ |w2·h̃| */}
        {pass.contrib.map((c, j) => {
          const dropped = showTrain && mask![j] === 0;
          return (
            <line
              key={`o${j}`}
              x1={XH + R}
              y1={yHid(j)}
              x2={XO - R - 2}
              y2={YO}
              stroke={dropped ? "var(--border)" : signedColor(c, maxContrib)}
              strokeWidth={dropped ? 1 : 1 + 4 * Math.min(1, Math.abs(c) / maxContrib)}
              strokeDasharray={dropped ? "3 3" : undefined}
            />
          );
        })}
        {/* inputs */}
        {net.x.map((v, i) => (
          <g key={`x${i}`}>
            <circle cx={XI} cy={yIn(i)} r={R} fill="var(--surface-2)" stroke="var(--text-muted)" strokeWidth={1} />
            <text x={XI} y={yIn(i) + 3.5} fontSize="9" textAnchor="middle" fill="var(--text)">
              {v.toFixed(2)}
            </text>
          </g>
        ))}
        <text x={XI} y={NH - 2} fontSize="9" textAnchor="middle" fill="var(--text-muted)">
          x
        </text>
        {/* hidden units */}
        {net.h.map((h, j) => {
          const dropped = showTrain && mask![j] === 0;
          const ht = pass.ht[j];
          return (
            <g key={`h${j}`} opacity={dropped ? 0.45 : 1}>
              <circle cx={XH} cy={yHid(j)} r={R} fill={dropped ? "var(--surface-2)" : "var(--surface)"} stroke={dropped ? "var(--text-faint)" : "var(--accent)"} strokeWidth={1.4} strokeDasharray={dropped ? "2 2" : undefined} />
              {dropped && (
                <g stroke="var(--clay)" strokeWidth={1.4}>
                  <line x1={XH - 5} y1={yHid(j) - 5} x2={XH + 5} y2={yHid(j) + 5} />
                  <line x1={XH - 5} y1={yHid(j) + 5} x2={XH + 5} y2={yHid(j) - 5} />
                </g>
              )}
              <text x={XH - R - 4} y={yHid(j) + 3.5} fontSize="9" textAnchor="end" fill="var(--text-muted)">
                {h.toFixed(2)}
              </text>
              <text x={XH + R + 4} y={yHid(j) - 3} fontSize="9" textAnchor="start" fill={dropped ? "var(--clay)" : "var(--text)"} fontFamily="ui-monospace, monospace">
                {dropped ? "→ 0" : `→ ${ht.toFixed(2)}`}
              </text>
            </g>
          );
        })}
        <text x={XH} y={NH - 0} fontSize="9" textAnchor="middle" fill="var(--text-muted)">
          h → h̃
        </text>
        {/* output */}
        <circle cx={XO} cy={YO} r={R + 6} fill="var(--surface)" stroke="var(--gold)" strokeWidth={2} />
        <text x={XO} y={YO + 3.5} fontSize="9.5" textAnchor="middle" fill="var(--text)" fontFamily="ui-monospace, monospace">
          {pass.y.toFixed(2)}
        </text>
        <text x={XO} y={YO + R + 20} fontSize="9" textAnchor="middle" fill="var(--text-muted)">
          {out === "linear" ? "y = w₂·h̃ + b₂" : "y = σ(w₂·h̃ + b₂)"}
        </text>
        <text x={XO} y={YO - R - 12} fontSize="9" textAnchor="middle" fill="var(--text-muted)">
          {showTrain ? `train, mask ${k}` : mode === "eval" ? "eval" : "eval (no mask yet)"}
        </text>
      </svg>

      <figure className="mt-3">
        <svg viewBox={`0 0 ${PW} ${PH}`} className="h-auto w-full" role="img" aria-label="Sampled train-mode outputs, their running mean, and the eval-mode output">
          {ticks.map((v, i) => (
            <g key={i}>
              <line x1={PP.l} x2={PW - PP.r} y1={py(v)} y2={py(v)} stroke="var(--border)" strokeWidth={0.6} />
              <text x={PP.l - 4} y={py(v) + 3} fontSize="9" textAnchor="end" fill="var(--text-muted)">
                {v.toFixed(2)}
              </text>
            </g>
          ))}
          {[0, 100, 200, 300].map((j) => (
            <text key={j} x={px(j)} y={PH - 6} fontSize="9" textAnchor="middle" fill="var(--text-muted)">
              {j}
            </text>
          ))}
          {full.ys.slice(0, k).map((v, j) => (
            <circle key={j} cx={px(j + 1)} cy={py(v)} r={j === k - 1 ? 3.6 : 1.8} fill={j === k - 1 ? "var(--gold)" : "var(--text-muted)"} opacity={j === k - 1 ? 1 : 0.5} />
          ))}
          {scheme === "classic" && <line x1={PP.l} x2={PW - PP.r} y1={py(naive)} y2={py(naive)} stroke="var(--clay)" strokeWidth={1.3} strokeDasharray="6 4" />}
          {nonlinear && <line x1={PP.l} x2={PW - PP.r} y1={py(exact)} y2={py(exact)} stroke="var(--text)" strokeWidth={1} strokeDasharray="2 3" />}
          <line x1={PP.l} x2={PW - PP.r} y1={py(evalPass.y)} y2={py(evalPass.y)} stroke="var(--gold)" strokeWidth={1.6} strokeDasharray="7 4" />
          {k > 0 && <path d={meanPath} fill="none" stroke="var(--accent)" strokeWidth={2} />}
        </svg>
        <figcaption className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted">
          <span>● train output for one mask (mask number on x-axis)</span>
          <span className="text-accent">— running mean</span>
          <span className="text-gold">– – eval-mode output</span>
          {nonlinear && <span>··· exact mask average</span>}
          {scheme === "classic" && <span className="text-clay">– – eval without the (1−p) factor (a bug)</span>}
        </figcaption>
      </figure>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[420px] text-xs">
          <thead>
            <tr className="text-left text-muted">
              <th className="py-1 pr-2 font-medium">unit</th>
              <th className="py-1 pr-2 font-medium">h</th>
              <th className="py-1 pr-2 font-medium">mask</th>
              <th className="py-1 pr-2 font-medium">h̃</th>
              <th className="py-1 pr-2 font-medium">w₂</th>
              <th className="py-1 font-medium">w₂·h̃</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {net.h.map((h, j) => {
              const dropped = showTrain && mask![j] === 0;
              return (
                <tr key={j} className={`border-t border-border ${dropped ? "text-faint" : ""}`}>
                  <td className="py-0.5 pr-2">{j + 1}</td>
                  <td className="py-0.5 pr-2">{h.toFixed(3)}</td>
                  <td className="py-0.5 pr-2">{showTrain ? mask![j] : "—"}</td>
                  <td className="py-0.5 pr-2">{pass.ht[j].toFixed(3)}</td>
                  <td className="py-0.5 pr-2">{net.w2[j].toFixed(2)}</td>
                  <td className="py-0.5">{pass.contrib[j].toFixed(3)}</td>
                </tr>
              );
            })}
            <tr className="border-t border-border">
              <td className="py-0.5 pr-2 font-sans text-muted" colSpan={5}>
                z = b₂ ({net.b2}) + Σ w₂·h̃
              </td>
              <td className="py-0.5">{pass.z.toFixed(3)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </SimShell>
  );
}
