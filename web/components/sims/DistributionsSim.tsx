"use client";

import { useEffect, useId, useMemo, useState, type ReactNode } from "react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import {
  DEFAULT_PARAMS,
  curvature,
  density,
  densityCurves,
  drawSamples,
  gradLogLik,
  gradLogPrior,
  histogram,
  isDiscrete,
  logLik,
  makeDataset,
  mapClosedForm,
  mleClosedForm,
  objective,
  paramRange,
  plotRange,
  prefixSums,
  runAscent,
  runningStats,
  target,
  trueMean,
  trueVariance,
  type DistKind,
  type DistParams,
  type MleSetup,
  type ModelKind,
} from "@/lib/sim/distributions";

type Tab = "dist" | "mle";

export default function DistributionsSim() {
  const [tab, setTab] = useState<Tab>("dist");
  const switcher = (
    <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Simulator mode">
      {(
        [
          ["dist", "Distributions & sampling"],
          ["mle", "Maximum likelihood / MAP"],
        ] as const
      ).map(([id, label]) => (
        <button key={id} role="tab" aria-selected={tab === id} className={`sim-btn ${tab === id ? "primary" : ""}`} onClick={() => setTab(id)}>
          {label}
        </button>
      ))}
    </div>
  );
  return tab === "dist" ? <DistributionsPanel switcher={switcher} /> : <MlePanel switcher={switcher} />;
}

/* ================================================================== */
/* Tab A — distributions, sampling, law of large numbers              */
/* ================================================================== */

const MAX_SAMPLES = 5000;
const DIST_LABELS: Record<DistKind, string> = { bernoulli: "Bernoulli", binomial: "Binomial", gaussian: "Gaussian", uniform: "Uniform" };

const AW = 560;
const AH = 260;
const AM = { l: 44, r: 12, t: 12, b: 30 };

function niceTicks(lo: number, hi: number, count = 6) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const stepT = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out: number[] = [];
  for (let t = Math.ceil(lo / stepT) * stepT; t <= hi + 1e-9; t += stepT) out.push(+t.toFixed(10));
  return out;
}

function DistributionsPanel({ switcher }: { switcher: ReactNode }) {
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const [kind, setKind] = useState<DistKind>("gaussian");
  const [q, setQ] = useState<DistParams>(DEFAULT_PARAMS);
  const [batch, setBatch] = useState(1);
  const [seed, setSeed] = useState(1);

  const samples = useMemo(() => drawSamples(kind, q, MAX_SAMPLES, seed), [kind, q, seed]);
  const sums = useMemo(() => prefixSums(samples), [samples]);
  const s = useStepper(MAX_SAMPLES / batch + 1, { baseMs: 300 });
  const { setStep } = s;
  useEffect(() => setStep(0), [kind, q, seed, batch, setStep]);

  const N = Math.min(MAX_SAMPLES, s.step * batch);
  const stats = runningStats(sums, N);
  const mu = trueMean(kind, q);
  const v = trueVariance(kind, q);
  const hist = useMemo(() => histogram(kind, q, samples, N), [kind, q, samples, N]);
  const discrete = isDiscrete(kind);
  const latest = N > 0 ? samples[N - 1] : null;

  const set = (patch: Partial<DistParams>) => setQ((old) => ({ ...old, ...patch }));

  /* ---------- main plot scales ---------- */
  const [xa, xb] = plotRange(kind, q);
  const PW = AW - AM.l - AM.r;
  const PH = AH - AM.t - AM.b;
  const sx = (x: number) => AM.l + ((x - xa) / (xb - xa)) * PW;

  const truthCurve = useMemo(() => {
    if (discrete) return [];
    const pts: [number, number][] = [];
    for (let i = 0; i <= 300; i++) {
      const x = xa + ((xb - xa) * i) / 300;
      pts.push([x, density(kind, x, q)]);
    }
    return pts;
  }, [discrete, kind, q, xa, xb]);

  const trueMax = discrete
    ? Math.max(...hist.edges.slice(0, -1).map((e) => density(kind, e + 0.5, q)))
    : Math.max(...truthCurve.map((p) => p[1]));
  const histMax = Math.max(0, ...hist.heights);
  const yMax = Math.max(trueMax * 1.2, Math.min(histMax, trueMax * 2)) * 1.05 || 1;
  const sy = (y: number) => AM.t + PH - (Math.min(y, yMax) / yMax) * PH;

  /* ---------- running-mean plot ---------- */
  const RH = 150;
  const RM = { l: 44, r: 12, t: 10, b: 26 };
  const RPW = AW - RM.l - RM.r;
  const RPH = RH - RM.t - RM.b;
  const sd = Math.sqrt(v);
  const runPts = useMemo(() => {
    if (N < 1) return [];
    const pts: [number, number][] = [];
    const stride = Math.max(1, Math.floor(N / 240));
    for (let i = 1; i <= N; i += stride) pts.push([i, sums.s1[i] / i]);
    if (pts[pts.length - 1][0] !== N) pts.push([N, sums.s1[N] / N]);
    return pts;
  }, [N, sums]);
  const halfSpan = Math.max(sd * 1.2, 1e-3);
  const ry = (y: number) => RM.t + RPH / 2 - (Math.max(-halfSpan, Math.min(halfSpan, y - mu)) / halfSpan) * (RPH / 2);
  const rx = (i: number) => RM.l + (N <= 1 ? 0 : ((i - 1) / (N - 1)) * RPW);
  const band = useMemo(() => {
    if (N < 2) return "";
    const up: string[] = [];
    const dn: string[] = [];
    for (let j = 0; j <= 80; j++) {
      const i = 1 + ((N - 1) * j) / 80;
      const e = (2 * sd) / Math.sqrt(i);
      up.push(`${rx(i).toFixed(1)},${ry(mu + e).toFixed(1)}`);
      dn.push(`${rx(i).toFixed(1)},${ry(mu - e).toFixed(1)}`);
    }
    return [...up, ...dn.reverse()].join(" ");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [N, sd, mu]);

  const se = sd / Math.sqrt(Math.max(1, N));
  const err = Math.abs(stats.mean - mu);

  let story: ReactNode;
  if (N === 0) {
    story = (
      <p>
        No samples yet. The gold {discrete ? "markers show the PMF" : "curve is the PDF"}: the distribution we will draw from. Press <b>Play</b> or <b>Step</b>. Each step draws {batch === 1 ? "one sample" : `${batch} samples`} from a seeded random generator and adds {batch === 1 ? "it" : "them"} to the histogram.
      </p>
    );
  } else {
    story = (
      <>
        <p>
          Sample #{N}: x = {discrete ? latest : fmt(latest ?? NaN)}. The bars show what share of the {N} samples landed in each {discrete ? "value" : "bin (scaled to a density)"}.
        </p>
        <p className="mt-1">
          Sample mean {fmt(stats.mean)} vs true mean {fmt(mu)}: error {fmt(err)}. By the law of large numbers this error shrinks as you sample more, roughly like σ/√N = {fmt(se)}
          {err <= 2 * se ? ". It is within two standard errors, which we expect about 95% of the time." : ". Right now it is outside two standard errors. That happens about 5% of the time, so keep sampling."}
        </p>
        {N < 30 && <p className="mt-1">With so few samples the histogram is still bumpy. It only looks like the {discrete ? "PMF" : "PDF"} after hundreds of draws.</p>}
      </>
    );
  }

  const formula: Record<DistKind, string> = {
    bernoulli: "P(X=1)=p, P(X=0)=1−p;  E[X]=p,  Var[X]=p(1−p)",
    binomial: "P(X=k)=C(n,k)·pᵏ(1−p)ⁿ⁻ᵏ;  E[X]=np,  Var[X]=np(1−p)",
    gaussian: "p(x)=exp(−(x−μ)²/2σ²)/√(2πσ²);  E[X]=μ,  Var[X]=σ²",
    uniform: "p(x)=1/(b−a) on [a,b];  E[X]=(a+b)/2,  Var[X]=(b−a)²/12",
  };

  return (
    <SimShell
      title="Probability Distributions & Sampling"
      subtitle="Draw random samples and watch the histogram settle into the true distribution."
      controls={
        <div className="space-y-2.5">
          {switcher}
          <Controls s={s} />
          <div className="flex flex-wrap items-center gap-1.5">
            {(Object.keys(DIST_LABELS) as DistKind[]).map((k) => (
              <button key={k} className={`sim-btn ${k === kind ? "primary" : ""}`} onClick={() => setKind(k)} aria-pressed={k === kind}>
                {DIST_LABELS[k]}
              </button>
            ))}
            <span className="ml-2 text-xs text-muted">samples/step</span>
            {[1, 10, 100].map((b) => (
              <button key={b} className={`sim-btn ${b === batch ? "primary" : ""}`} onClick={() => setBatch(b)} aria-pressed={b === batch}>
                {b}
              </button>
            ))}
            <button className="sim-btn" onClick={() => setSeed((x) => x + 1)} aria-label="New random seed">
              New seed ({seed})
            </button>
          </div>
          <div className="grid gap-1.5 sm:grid-cols-2">
            {(kind === "bernoulli" || kind === "binomial") && <Slider label="p" value={q.p} min={0} max={1} step={0.01} onChange={(p) => set({ p })} />}
            {kind === "binomial" && <Slider label="n (trials)" value={q.n} min={1} max={30} step={1} onChange={(n) => set({ n })} fmt={(x) => String(x)} />}
            {kind === "gaussian" && <Slider label="mean μ" value={q.mu} min={-3} max={3} step={0.1} onChange={(m) => set({ mu: m })} />}
            {kind === "gaussian" && <Slider label="std σ" value={q.sigma} min={0.3} max={3} step={0.05} onChange={(sg) => set({ sigma: sg })} />}
            {kind === "uniform" && <Slider label="lower a" value={q.a} min={-4} max={3} step={0.1} onChange={(a) => set({ a, b: Math.max(q.b, a + 0.5) })} />}
            {kind === "uniform" && <Slider label="upper b" value={q.b} min={-3.5} max={4} step={0.1} onChange={(b) => set({ b, a: Math.min(q.a, b - 0.5) })} />}
          </div>
        </div>
      }
      aside={
        <div className="space-y-3">
          <Narration title={N === 0 ? "Ready to sample" : `After ${N} sample${N === 1 ? "" : "s"}`}>{story}</Narration>
          <div className="rounded-lg border border-border p-3 text-xs">
            <div className="mb-1 font-medium">{DIST_LABELS[kind]}</div>
            <p className="font-mono text-muted">{formula[kind]}</p>
            <table className="mt-2 w-full font-mono">
              <thead>
                <tr className="text-muted">
                  <th className="text-left font-normal" />
                  <th className="text-right font-normal">sample</th>
                  <th className="text-right font-normal">true</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="text-muted">mean</td>
                  <td className="text-right">{N ? fmt(stats.mean) : "—"}</td>
                  <td className="text-right">{fmt(mu)}</td>
                </tr>
                <tr>
                  <td className="text-muted">variance</td>
                  <td className="text-right">{N ? fmt(stats.variance) : "—"}</td>
                  <td className="text-right">{fmt(v)}</td>
                </tr>
                <tr>
                  <td className="text-muted">std error</td>
                  <td className="text-right" colSpan={2}>
                    σ/√N = {N ? fmt(se) : "—"}
                  </td>
                </tr>
              </tbody>
            </table>
            <p className="mt-1.5 text-muted">Sample variance here divides by N (the maximum-likelihood estimate).</p>
          </div>
        </div>
      }
    >
      <svg viewBox={`0 0 ${AW} ${AH}`} className="h-auto w-full" role="img" aria-label={`Histogram of ${N} samples from a ${DIST_LABELS[kind]} distribution, with the true ${discrete ? "PMF" : "PDF"} overlaid`}>
        <defs>
          <clipPath id={`clip-${uid}`}>
            <rect x={AM.l} y={AM.t - 2} width={PW} height={PH + 2} />
          </clipPath>
        </defs>
        {niceTicks(0, yMax, 4).map((t) => (
          <g key={`y${t}`}>
            <line x1={AM.l} x2={AW - AM.r} y1={sy(t)} y2={sy(t)} stroke="var(--border)" strokeWidth={0.7} />
            <text x={AM.l - 5} y={sy(t) + 3} textAnchor="end" fontSize={9} fill="var(--text-muted)">
              {+t.toFixed(3)}
            </text>
          </g>
        ))}
        <line x1={AM.l} x2={AW - AM.r} y1={AM.t + PH} y2={AM.t + PH} stroke="var(--text-muted)" />
        {(discrete ? hist.edges.slice(0, -1).map((e) => e + 0.5).filter((_, i, arr) => arr.length <= 16 || i % Math.ceil(arr.length / 16) === 0) : niceTicks(xa, xb, 8)).map((t) => (
          <text key={`x${t}`} x={sx(t)} y={AH - 12} textAnchor="middle" fontSize={9} fill="var(--text-muted)">
            {+t.toFixed(2)}
          </text>
        ))}
        <text x={AM.l + PW / 2} y={AH - 1} textAnchor="middle" fontSize={9} fill="var(--text-muted)">
          x
        </text>
        <text x={10} y={AM.t + PH / 2} fontSize={9} fill="var(--text-muted)" transform={`rotate(-90 10 ${AM.t + PH / 2})`} textAnchor="middle">
          {discrete ? "probability" : "density"}
        </text>
        <g clipPath={`url(#clip-${uid})`}>
          {hist.heights.map((h, i) => {
            const x0 = sx(hist.edges[i]);
            const x1 = sx(hist.edges[i + 1]);
            const pad = discrete ? (x1 - x0) * 0.18 : 0.5;
            return h > 0 ? <rect key={i} x={x0 + pad} y={sy(h)} width={Math.max(0.5, x1 - x0 - 2 * pad)} height={AM.t + PH - sy(h)} fill="color-mix(in srgb, var(--accent) 45%, transparent)" stroke="var(--accent)" strokeWidth={0.6} /> : null;
          })}
          {discrete
            ? hist.edges.slice(0, -1).map((e) => {
                const k = e + 0.5;
                const pk = density(kind, k, q);
                const w = (sx(1) - sx(0)) * 0.36;
                return <line key={k} x1={sx(k) - w} x2={sx(k) + w} y1={sy(pk)} y2={sy(pk)} stroke="var(--gold)" strokeWidth={3} />;
              })
            : <polyline points={truthCurve.map(([x, y]) => `${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(" ")} fill="none" stroke="var(--gold)" strokeWidth={2.2} />}
          {latest !== null && (
            <path d={`M ${sx(latest)} ${AM.t + PH} l -5 9 l 10 0 z`} fill="var(--clay)" aria-hidden="true" />
          )}
          <line x1={sx(mu)} x2={sx(mu)} y1={AM.t} y2={AM.t + PH} stroke="var(--text-muted)" strokeDasharray="3 3" />
          {N > 0 && <line x1={sx(stats.mean)} x2={sx(stats.mean)} y1={AM.t} y2={AM.t + PH} stroke="var(--accent)" strokeWidth={1.5} />}
        </g>
        <g fontSize={9}>
          <rect x={AW - AM.r - 182} y={AM.t + 2} width={10} height={8} fill="color-mix(in srgb, var(--accent) 45%, transparent)" stroke="var(--accent)" />
          <text x={AW - AM.r - 168} y={AM.t + 10} fill="var(--text-muted)">samples</text>
          <line x1={AW - AM.r - 120} x2={AW - AM.r - 108} y1={AM.t + 6} y2={AM.t + 6} stroke="var(--gold)" strokeWidth={2.5} />
          <text x={AW - AM.r - 104} y={AM.t + 10} fill="var(--text-muted)">{discrete ? "true PMF" : "true PDF"}</text>
          <line x1={AW - AM.r - 50} x2={AW - AM.r - 40} y1={AM.t + 6} y2={AM.t + 6} stroke="var(--accent)" strokeWidth={1.5} />
          <text x={AW - AM.r - 36} y={AM.t + 10} fill="var(--text-muted)">x̄</text>
        </g>
      </svg>

      <div className="mt-2 text-xs text-muted">Running sample mean (accent) vs true mean (dashed). Shaded band = true mean ± 2σ/√n.</div>
      <svg viewBox={`0 0 ${AW} ${RH}`} className="h-auto w-full" role="img" aria-label="Running sample mean as more samples are drawn, approaching the true mean">
        <rect x={RM.l} y={RM.t} width={RPW} height={RPH} fill="none" stroke="var(--border)" />
        {band && <polygon points={band} fill="color-mix(in srgb, var(--accent) 12%, transparent)" stroke="none" />}
        <line x1={RM.l} x2={RM.l + RPW} y1={ry(mu)} y2={ry(mu)} stroke="var(--text-muted)" strokeDasharray="4 3" />
        {runPts.length > 1 && <polyline points={runPts.map(([i, m]) => `${rx(i).toFixed(1)},${ry(m).toFixed(1)}`).join(" ")} fill="none" stroke="var(--accent)" strokeWidth={1.6} />}
        {runPts.length > 0 && <circle cx={rx(N)} cy={ry(stats.mean)} r={3.5} fill="var(--gold)" />}
        <text x={RM.l - 5} y={ry(mu) + 3} textAnchor="end" fontSize={9} fill="var(--text-muted)">
          {+mu.toFixed(2)}
        </text>
        <text x={RM.l - 5} y={RM.t + 8} textAnchor="end" fontSize={9} fill="var(--text-muted)">
          {+(mu + halfSpan).toFixed(2)}
        </text>
        <text x={RM.l - 5} y={RM.t + RPH} textAnchor="end" fontSize={9} fill="var(--text-muted)">
          {+(mu - halfSpan).toFixed(2)}
        </text>
        <text x={RM.l} y={RH - 8} fontSize={9} fill="var(--text-muted)">
          n = 1
        </text>
        <text x={RM.l + RPW} y={RH - 8} textAnchor="end" fontSize={9} fill="var(--text-muted)">
          n = {N}
        </text>
        {N === 0 && (
          <text x={RM.l + RPW / 2} y={RM.t + RPH / 2 - 6} textAnchor="middle" fontSize={10} fill="var(--text-muted)">
            draw samples to see the running mean
          </text>
        )}
      </svg>
    </SimShell>
  );
}

/* ================================================================== */
/* Tab B — maximum likelihood and MAP by gradient ascent              */
/* ================================================================== */

const N_ASCENT = 60;
const SIGMA_KNOWN = 1.5;
const BW = 560;
const BH = 250;
const BM = { l: 52, r: 12, t: 12, b: 34 };
const DH = 170;
const DM = { l: 52, r: 12, t: 10, b: 26 };

const MODEL_DEFAULTS: Record<ModelKind, { truth: number; theta0: number; eta: number }> = {
  "gaussian-mean": { truth: 2, theta0: -2.5, eta: 1 },
  "bernoulli-p": { truth: 0.7, theta0: 0.2, eta: 0.05 },
};

function MlePanel({ switcher }: { switcher: ReactNode }) {
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const [model, setModel] = useState<ModelKind>("gaussian-mean");
  const [n, setN] = useState(10);
  const [truth, setTruth] = useState(MODEL_DEFAULTS["gaussian-mean"].truth);
  const [seed, setSeed] = useState(1);
  const [useMap, setUseMap] = useState(false);
  const [mu0, setMu0] = useState(0);
  const [tau, setTau] = useState(1);
  const [alpha, setAlpha] = useState(2);
  const [beta, setBeta] = useState(2);
  const [theta0, setTheta0] = useState(MODEL_DEFAULTS["gaussian-mean"].theta0);
  const [eta, setEta] = useState(MODEL_DEFAULTS["gaussian-mean"].eta);

  const gaussianModel = model === "gaussian-mean";
  const chooseModel = (m: ModelKind) => {
    setModel(m);
    setTruth(MODEL_DEFAULTS[m].truth);
    setTheta0(MODEL_DEFAULTS[m].theta0);
    setEta(MODEL_DEFAULTS[m].eta);
  };

  const data = useMemo(() => makeDataset(model, n, truth, SIGMA_KNOWN, seed), [model, n, truth, seed]);
  const setup: MleSetup = useMemo(() => ({ model, data, sigma: SIGMA_KNOWN, useMap, mu0, tau, alpha, beta }), [model, data, useMap, mu0, tau, alpha, beta]);
  const traj = useMemo(() => runAscent(setup, theta0, eta, N_ASCENT), [setup, theta0, eta]);
  const s = useStepper(N_ASCENT + 1, { baseMs: 700 });
  const { setStep } = s;
  useEffect(() => setStep(0), [traj, setStep]);

  const k = s.step;
  const pending = k < traj.length ? traj[k] : null;
  const thCur = pending ? pending.theta : traj[traj.length - 1].thetaNext;
  const objCur = objective(setup, thCur);
  const gCur = gradLogLik(setup, thCur) + (useMap ? gradLogPrior(setup, thCur) : 0);
  const mle = mleClosedForm(setup);
  const mapEst = mapClosedForm(setup);
  const goal = target(setup);
  const curv = curvature(setup, goal);
  const etaStable = 2 / curv;
  const heads = data.reduce((a, b) => a + b, 0);

  /* ---------- objective plot ---------- */
  const [ta, tb] = paramRange(setup);
  const PW = BW - BM.l - BM.r;
  const PH = BH - BM.t - BM.b;
  const sx = (t: number) => BM.l + ((t - ta) / (tb - ta)) * PW;
  const thView = Math.max(ta, Math.min(tb, thCur)); // keep the marker visible if the iterates diverge

  const curves = useMemo(() => {
    const pts = 260;
    const xs = Array.from({ length: pts }, (_, i) => ta + ((tb - ta) * i) / (pts - 1));
    return { xs, ll: xs.map((t) => logLik(setup, t)), obj: xs.map((t) => objective(setup, t)) };
  }, [setup, ta, tb]);
  const allY = [...curves.obj, ...(useMap ? curves.ll : [])].filter(Number.isFinite);
  const yTop = Math.max(...allY);
  const yLow = Math.max(Math.min(...allY), yTop - Math.max(8, (gaussianModel ? 6 : 3) * n));
  const yHi = yTop + (yTop - yLow) * 0.08;
  const sy = (y: number) => BM.t + ((yHi - Math.max(yLow, Math.min(yHi, y))) / (yHi - yLow)) * PH;
  const line = (ys: number[]) => curves.xs.map((x, i) => `${sx(x).toFixed(1)},${sy(ys[i]).toFixed(1)}`).join(" ");

  const tanHalf = (tb - ta) * 0.08;
  const visited = [...traj.slice(0, k).map((st) => st.theta), thCur];

  /* ---------- density plot ---------- */
  const dens = useMemo(() => densityCurves(setup), [setup]);
  const dMax = Math.max(...dens.likelihood, ...(useMap ? [...dens.prior, ...dens.posterior] : [])) * 1.08 || 1;
  const DPH = DH - DM.t - DM.b;
  const dy = (y: number) => DM.t + DPH - (y / dMax) * DPH;
  const dLine = (ys: number[]) => dens.xs.map((x, i) => `${sx(x).toFixed(1)},${dy(ys[i]).toFixed(1)}`).join(" ");

  const paramName = gaussianModel ? "μ" : "p";
  const objName = useMap ? "log-posterior ℓ(θ) + log p(θ)" : "log-likelihood ℓ(θ)";

  let story: ReactNode;
  if (pending) {
    story = (
      <>
        <p>
          Step {k + 1}: at {paramName} = {fmt(pending.theta, 4)} the {useMap ? "log-posterior" : "log-likelihood"} is {fmt(pending.obj, 3)} and its derivative is {fmt(pending.grad, 4)} ({pending.grad > 0 ? `uphill is to the right, so ${paramName} goes up` : pending.grad < 0 ? `uphill is to the left, so ${paramName} goes down` : "zero: a flat spot"}).
        </p>
        <p className="mt-1 font-mono text-xs text-text">
          {paramName} ← {fmt(pending.theta, 4)} + {fmt(eta)} × ({fmt(pending.grad, 4)} / {n}) = {fmt(pending.thetaNext, 4)}
        </p>
        <p className="mt-1">
          This is gradient <b>ascent</b>: we step along the gradient because we want the maximum. Dividing by n = {n} uses the average log-likelihood, just like a mean loss in training.
        </p>
      </>
    );
  } else {
    const gap = Math.abs(thCur - goal);
    story = (
      <p>
        After {N_ASCENT} steps, {paramName} = {fmt(thCur, 4)}. The exact formula gives the {useMap ? "MAP" : "maximum-likelihood"} estimate {fmt(goal, 4)} (gap {fmt(gap, 5)}).{" "}
        {gap < 1e-3 ? "Gradient ascent found the same answer as the formula." : eta >= etaStable ? "The step size is above the safe limit, so the steps bounce around or blow up. Lower η." : "Not there yet. It would need a bigger η or more steps."}
      </p>
    );
  }

  return (
    <SimShell
      title="Maximum Likelihood & MAP"
      subtitle={gaussianModel ? `Data xᵢ ~ N(μ_true, σ²), with σ = ${SIGMA_KNOWN} known. Estimate μ.` : "Coin flips xᵢ ~ Bernoulli(p_true). Estimate p."}
      controls={
        <div className="space-y-2.5">
          {switcher}
          <Controls s={s} />
          <div className="flex flex-wrap items-center gap-1.5">
            <button className={`sim-btn ${gaussianModel ? "primary" : ""}`} onClick={() => chooseModel("gaussian-mean")} aria-pressed={gaussianModel}>
              Gaussian mean μ
            </button>
            <button className={`sim-btn ${!gaussianModel ? "primary" : ""}`} onClick={() => chooseModel("bernoulli-p")} aria-pressed={!gaussianModel}>
              Coin flips p
            </button>
            <button className="sim-btn" onClick={() => setSeed((x) => x + 1)} aria-label="Resample the dataset">
              Resample data
            </button>
            <label className="ml-1 flex items-center gap-1.5 text-xs">
              <input type="checkbox" checked={useMap} onChange={(e) => setUseMap(e.target.checked)} aria-label="Add a prior (MAP estimation)" />
              MAP: add a {gaussianModel ? "Gaussian" : "Beta"} prior
            </label>
          </div>
          <div className="grid gap-1.5 sm:grid-cols-2">
            <Slider label="data size n" value={n} min={1} max={200} step={1} onChange={setN} fmt={(x) => String(x)} />
            <Slider label={gaussianModel ? "true μ" : "true p"} value={truth} min={gaussianModel ? -2 : 0.05} max={gaussianModel ? 4 : 0.95} step={gaussianModel ? 0.1 : 0.01} onChange={setTruth} />
            <Slider label={`start ${paramName}₀`} value={theta0} min={gaussianModel ? -4 : 0.01} max={gaussianModel ? 6 : 0.99} step={0.01} onChange={setTheta0} />
            <Slider label="step size η" value={eta} min={gaussianModel ? 0.01 : 0.005} max={gaussianModel ? 5 : 0.3} step={gaussianModel ? 0.01 : 0.005} onChange={setEta} fmt={(x) => x.toFixed(3)} />
            {useMap && gaussianModel && <Slider label="prior mean μ₀" value={mu0} min={-3} max={4} step={0.1} onChange={setMu0} />}
            {useMap && gaussianModel && <Slider label="prior std τ" value={tau} min={0.3} max={3} step={0.05} onChange={setTau} />}
            {useMap && !gaussianModel && <Slider label="prior α" value={alpha} min={1} max={30} step={0.5} onChange={setAlpha} fmt={(x) => x.toFixed(1)} />}
            {useMap && !gaussianModel && <Slider label="prior β" value={beta} min={1} max={30} step={0.5} onChange={setBeta} fmt={(x) => x.toFixed(1)} />}
          </div>
        </div>
      }
      aside={
        <div className="space-y-3">
          <Narration title={pending ? `Ascent step ${k + 1}` : "Did it converge?"}>{story}</Narration>
          <div className="rounded-lg border border-border p-3 text-xs">
            <div className="mb-1 font-medium">Exact formula answers</div>
            {gaussianModel ? (
              <p className="text-muted">
                MLE μ̂ = x̄ = <span className="font-mono text-text">{fmt(mle, 4)}</span>
                <br />
                MAP = (Σxᵢ/σ² + μ₀/τ²)/(n/σ² + 1/τ²) = <span className="font-mono text-text">{fmt(mapEst, 4)}</span>
              </p>
            ) : (
              <p className="text-muted">
                k = {heads} heads in n = {n}. MLE p̂ = k/n = <span className="font-mono text-text">{fmt(mle, 4)}</span>
                <br />
                MAP = (k+α−1)/(n+α+β−2) = <span className="font-mono text-text">{fmt(mapEst, 4)}</span>
              </p>
            )}
            <p className="mt-1.5 text-muted">
              Current {paramName} = <span className="font-mono text-text">{fmt(thCur, 4)}</span>, objective = <span className="font-mono text-text">{fmt(objCur, 3)}</span>, gradient = <span className="font-mono text-text">{fmt(gCur, 4)}</span>
            </p>
            <p className="mt-1.5 text-muted">
              Curvature (how sharply the averaged objective bends) at the peak ≈ {fmt(curv)}. Ascent is stable for η &lt; 2/curvature = {fmt(etaStable)}. Your η = {fmt(eta)} is{" "}
              <span className={eta < etaStable ? "text-accent" : "text-clay"}>{eta < etaStable ? "below" : "above"}</span> it.
            </p>
            {useMap && <p className="mt-1.5 text-muted">With small n, the prior pulls the estimate toward {gaussianModel ? "μ₀" : "the prior mode"}. Raise n and the MAP estimate moves toward the MLE.</p>}
          </div>
        </div>
      }
    >
      <svg viewBox={`0 0 ${BW} ${BH}`} className="h-auto w-full" role="img" aria-label={`The ${objName} as a function of ${paramName}, with the current gradient-ascent iterate`}>
        <defs>
          <clipPath id={`clipm-${uid}`}>
            <rect x={BM.l} y={BM.t} width={PW} height={PH} />
          </clipPath>
        </defs>
        {niceTicks(yLow, yHi, 4).map((t) => (
          <g key={`y${t}`}>
            <line x1={BM.l} x2={BW - BM.r} y1={sy(t)} y2={sy(t)} stroke="var(--border)" strokeWidth={0.7} />
            <text x={BM.l - 5} y={sy(t) + 3} textAnchor="end" fontSize={9} fill="var(--text-muted)">
              {+t.toFixed(1)}
            </text>
          </g>
        ))}
        {niceTicks(ta, tb, 8).map((t) => (
          <text key={`x${t}`} x={sx(t)} y={BH - 18} textAnchor="middle" fontSize={9} fill="var(--text-muted)">
            {+t.toFixed(2)}
          </text>
        ))}
        <text x={BM.l + PW / 2} y={BH - 4} textAnchor="middle" fontSize={9} fill="var(--text-muted)">
          parameter {paramName}
        </text>
        <g clipPath={`url(#clipm-${uid})`}>
          {useMap && <polyline points={line(curves.ll)} fill="none" stroke="var(--text-muted)" strokeWidth={1.4} strokeDasharray="5 4" />}
          <polyline points={line(curves.obj)} fill="none" stroke="var(--accent)" strokeWidth={2.2} />
          <line x1={sx(goal)} x2={sx(goal)} y1={BM.t} y2={BM.t + PH} stroke="var(--accent)" strokeDasharray="3 3" />
          {gaussianModel &&
            data.slice(0, 200).map((x, i) => <line key={i} x1={sx(x)} x2={sx(x)} y1={BM.t + PH - 7} y2={BM.t + PH} stroke="var(--text-muted)" strokeWidth={1} opacity={0.7} />)}
          {visited.slice(0, -1).map((t, i) => (
            <circle key={i} cx={sx(t)} cy={sy(objective(setup, t))} r={2.3} fill="var(--text-muted)" opacity={0.6} />
          ))}
          {Number.isFinite(objCur) && Number.isFinite(gCur) && (
            <line x1={sx(thCur - tanHalf)} y1={sy(objCur - gCur * tanHalf)} x2={sx(thCur + tanHalf)} y2={sy(objCur + gCur * tanHalf)} stroke="var(--clay)" strokeWidth={2} />
          )}
          <circle cx={sx(thView)} cy={sy(objCur)} r={5} fill="var(--gold)" stroke="var(--text)" strokeWidth={0.8} />
        </g>
        <g fontSize={9}>
          <line x1={BM.l + 8} x2={BM.l + 22} y1={BM.t + 8} y2={BM.t + 8} stroke="var(--accent)" strokeWidth={2.2} />
          <text x={BM.l + 26} y={BM.t + 11} fill="var(--text-muted)">{useMap ? "log-posterior (unnormalised)" : "log-likelihood"}</text>
          {useMap && (
            <>
              <line x1={BM.l + 8} x2={BM.l + 22} y1={BM.t + 21} y2={BM.t + 21} stroke="var(--text-muted)" strokeDasharray="5 4" strokeWidth={1.4} />
              <text x={BM.l + 26} y={BM.t + 24} fill="var(--text-muted)">log-likelihood</text>
            </>
          )}
          <line x1={BM.l + 8} x2={BM.l + 22} y1={BM.t + (useMap ? 34 : 21)} y2={BM.t + (useMap ? 34 : 21)} stroke="var(--clay)" strokeWidth={2} />
          <text x={BM.l + 26} y={BM.t + (useMap ? 37 : 24)} fill="var(--text-muted)">tangent (slope = gradient)</text>
        </g>
      </svg>

      {!gaussianModel && (
        <div className="mt-1 flex flex-wrap items-center gap-1" aria-label={`${heads} heads out of ${n} flips`}>
          {data.slice(0, 100).map((x, i) => (
            <span key={i} className="inline-block h-2.5 w-2.5 rounded-full border border-border" style={{ background: x ? "var(--accent)" : "transparent" }} />
          ))}
          <span className="ml-1 text-xs text-muted">
            {heads} heads (filled) / {n} flips{n > 100 ? " (first 100 shown)" : ""}
          </span>
        </div>
      )}
      {gaussianModel && <div className="mt-1 text-xs text-muted">The ticks along the bottom are the {n} data points (same axis as μ).</div>}

      <div className="mt-3 text-xs text-muted">Normalised densities over {paramName}: likelihood{useMap ? ", prior and posterior" : ""}. Vertical lines: MLE (dashed){useMap ? ", MAP (solid accent)" : ""}, current guess (gold).</div>
      <svg viewBox={`0 0 ${BW} ${DH}`} className="h-auto w-full" role="img" aria-label={`Likelihood${useMap ? ", prior and posterior" : ""} as normalised densities over ${paramName}`}>
        <line x1={DM.l} x2={BW - DM.r} y1={DM.t + DPH} y2={DM.t + DPH} stroke="var(--text-muted)" />
        {niceTicks(ta, tb, 8).map((t) => (
          <text key={`dx${t}`} x={sx(t)} y={DH - 10} textAnchor="middle" fontSize={9} fill="var(--text-muted)">
            {+t.toFixed(2)}
          </text>
        ))}
        {useMap && <polyline points={dLine(dens.prior)} fill="none" stroke="var(--clay)" strokeWidth={1.8} strokeDasharray="6 4" />}
        <polyline points={dLine(dens.likelihood)} fill="none" stroke="var(--text-muted)" strokeWidth={1.8} />
        {useMap && <polyline points={dLine(dens.posterior)} fill="color-mix(in srgb, var(--accent) 14%, transparent)" stroke="var(--accent)" strokeWidth={2.2} />}
        <line x1={sx(mle)} x2={sx(mle)} y1={DM.t} y2={DM.t + DPH} stroke="var(--text-muted)" strokeDasharray="3 3" />
        {useMap && <line x1={sx(mapEst)} x2={sx(mapEst)} y1={DM.t} y2={DM.t + DPH} stroke="var(--accent)" strokeWidth={1.5} />}
        <line x1={sx(thView)} x2={sx(thView)} y1={DM.t} y2={DM.t + DPH} stroke="var(--gold)" strokeWidth={1.5} />
        <g fontSize={9}>
          <line x1={BW - DM.r - 196} x2={BW - DM.r - 184} y1={DM.t + 6} y2={DM.t + 6} stroke="var(--text-muted)" strokeWidth={1.8} />
          <text x={BW - DM.r - 180} y={DM.t + 9} fill="var(--text-muted)">likelihood</text>
          {useMap && (
            <>
              <line x1={BW - DM.r - 130} x2={BW - DM.r - 118} y1={DM.t + 6} y2={DM.t + 6} stroke="var(--clay)" strokeDasharray="6 4" strokeWidth={1.8} />
              <text x={BW - DM.r - 114} y={DM.t + 9} fill="var(--text-muted)">prior</text>
              <line x1={BW - DM.r - 80} x2={BW - DM.r - 68} y1={DM.t + 6} y2={DM.t + 6} stroke="var(--accent)" strokeWidth={2.2} />
              <text x={BW - DM.r - 64} y={DM.t + 9} fill="var(--text-muted)">posterior</text>
            </>
          )}
        </g>
      </svg>
    </SimShell>
  );
}
