"use client";

import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import {
  PRESETS,
  apply,
  blend,
  det,
  ease,
  eigen,
  svd,
  svdStage,
  trace,
  transpose,
  viewRadius,
  type Mat2,
  type Vec2,
} from "@/lib/sim/linearTransform";

type Mode = "transform" | "svd";

const SIZE = 400; // SVG viewBox is SIZE × SIZE, origin at the centre
const HALF = SIZE / 2;
const TWEEN_MS = 950;

const deg = (rad: number) => (rad * 180) / Math.PI;
const show = (v: number, d = 2) => fmt(Math.abs(v) < 5e-4 ? 0 : v, d);

/** A 2×2 matrix printed between brackets. */
function MatView({ name, M, note }: { name: string; M: Mat2; note?: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="font-serif text-sm font-semibold">{name}</span>
      <span className="text-muted">=</span>
      <div className="grid grid-cols-2 gap-x-2 rounded border-x-2 border-border-strong px-1.5 py-0.5 font-mono text-[0.78rem] leading-5" aria-label={`${name} = [[${show(M.a)}, ${show(M.b)}], [${show(M.c)}, ${show(M.d)}]]`}>
        <span className="text-right">{show(M.a)}</span>
        <span className="text-right">{show(M.b)}</span>
        <span className="text-right">{show(M.c)}</span>
        <span className="text-right">{show(M.d)}</span>
      </div>
      {note && <span className="text-xs text-faint">{note}</span>}
    </div>
  );
}

/** Arrow from the origin-side point p to q, in screen coordinates, with a filled head. */
function Arrow({ p, q, color, width = 2.5, dashed = false, label }: { p: [number, number]; q: [number, number]; color: string; width?: number; dashed?: boolean; label?: string }) {
  const dx = q[0] - p[0];
  const dy = q[1] - p[1];
  const len = Math.hypot(dx, dy);
  if (len < 1e-6) return <circle cx={q[0]} cy={q[1]} r={3} fill={color} />;
  const ux = dx / len;
  const uy = dy / len;
  const head = Math.min(10, len * 0.45);
  const bx = q[0] - ux * head;
  const by = q[1] - uy * head;
  const w = head * 0.5;
  return (
    <g>
      <line x1={p[0]} y1={p[1]} x2={bx} y2={by} stroke={color} strokeWidth={width} strokeDasharray={dashed ? "5 4" : undefined} strokeLinecap="round" />
      <polygon points={`${q[0]},${q[1]} ${bx - uy * w},${by + ux * w} ${bx + uy * w},${by - ux * w}`} fill={color} opacity={dashed ? 0.7 : 1} />
      {label && (
        <text x={q[0] + ux * 8 + 2} y={q[1] + uy * 8 + 4} fontSize={11.5} fill={color} fontWeight={600} style={{ paintOrder: "stroke", stroke: "var(--surface)", strokeWidth: 3 }}>
          {label}
        </text>
      )}
    </g>
  );
}

export default function LinearTransformSim() {
  const [A, setA] = useState<Mat2>(PRESETS[0].M);
  const [preset, setPreset] = useState<string>(PRESETS[0].key);
  const [mode, setMode] = useState<Mode>("transform");
  const [x, setX] = useState<Vec2>([1, 1]);
  const [dragging, setDragging] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);

  const total = mode === "transform" ? 2 : 4;
  const s = useStepper(total, { baseMs: 1700 });
  const { setStep, setPlaying, reset } = s;

  /* ---- smooth tween of the continuous position towards the discrete step ---- */
  const [pos, setPos] = useState(0);
  const posRef = useRef(0);
  const rafRef = useRef(0);
  useEffect(() => {
    const from = posRef.current;
    const to = s.step;
    if (Math.abs(from - to) < 1e-9) return;
    const dur = TWEEN_MS / s.speed;
    const t0 = performance.now();
    const tick = (now: number) => {
      const u = Math.min(1, (now - t0) / dur);
      const p = from + (to - from) * ease(u);
      posRef.current = p;
      setPos(p);
      if (u < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [s.step, s.speed]);

  const jumpTo = (step: number) => {
    cancelAnimationFrame(rafRef.current);
    setPlaying(false);
    posRef.current = step;
    setPos(step);
    setStep(step);
  };

  const changeMode = (m: Mode) => {
    if (m === mode) return;
    setMode(m);
    cancelAnimationFrame(rafRef.current);
    posRef.current = 0;
    setPos(0);
    reset();
  };

  /** Editing the matrix jumps to the fully transformed state so the edit is visible at once. */
  const editA = (next: Mat2, presetKey = "custom") => {
    setA(next);
    setPreset(presetKey);
    jumpTo(total - 1);
  };

  /* ---- the real computations ---- */
  const dec = useMemo(() => svd(A), [A]);
  const eig = useMemo(() => eigen(A), [A]);
  const detA = det(A);
  const R = useMemo(() => viewRadius(A), [A]);
  const k = HALF / R; // pixels per unit
  const M: Mat2 = mode === "transform" ? blend(A, pos) : svdStage(dec, pos);
  const detM = det(M);
  const Mx = apply(M, x);

  const scr = (v: Vec2): [number, number] => [HALF + v[0] * k, HALF - v[1] * k];
  const O = scr([0, 0]);

  /* ---- geometry ---- */
  const G = Math.ceil(R * 1.6);
  const gridLines = useMemo(() => {
    const out: { p: Vec2; q: Vec2; axis: boolean }[] = [];
    for (let i = -G; i <= G; i++) {
      out.push({ p: apply(M, [i, -G]), q: apply(M, [i, G]), axis: i === 0 });
      out.push({ p: apply(M, [-G, i]), q: apply(M, [G, i]), axis: i === 0 });
    }
    return out;
  }, [M, G]);

  const circlePath = (T: Mat2) => {
    const pts: string[] = [];
    for (let i = 0; i <= 96; i++) {
      const th = (2 * Math.PI * i) / 96;
      const p = scr(apply(T, [Math.cos(th), Math.sin(th)]));
      pts.push(`${i === 0 ? "M" : "L"}${p[0].toFixed(2)},${p[1].toFixed(2)}`);
    }
    return pts.join(" ") + " Z";
  };

  const square = ([[0, 0], [1, 0], [1, 1], [0, 1]] as Vec2[]).map((v) => scr(apply(M, v)).join(",")).join(" ");
  const areaColor = detM < -1e-9 ? "var(--clay)" : "var(--accent)";

  /* ---- pointer interaction: drag the input vector x ---- */
  const toWorld = (e: ReactPointerEvent<SVGElement>): Vec2 | null => {
    const svg = svgRef.current;
    if (!svg) return null;
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * SIZE;
    const py = ((e.clientY - r.top) / r.height) * SIZE;
    const wx = Math.round(((px - HALF) / k) * 10) / 10;
    const wy = Math.round(((HALF - py) / k) * 10) / 10;
    return [Math.max(-R, Math.min(R, wx)), Math.max(-R, Math.min(R, wy))];
  };
  const onDown = (e: ReactPointerEvent<SVGSVGElement>) => {
    const w = toWorld(e);
    if (!w) return;
    setX(w);
    setDragging(true);
    svgRef.current?.setPointerCapture(e.pointerId);
  };
  const onMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!dragging) return;
    const w = toWorld(e);
    if (w) setX(w);
  };
  const onUp = (e: ReactPointerEvent<SVGSVGElement>) => {
    setDragging(false);
    if (svgRef.current?.hasPointerCapture(e.pointerId)) svgRef.current.releasePointerCapture(e.pointerId);
  };

  /* ---- eigen information (for A) ---- */
  const eigenLines: { v: Vec2; lam: number; idx: number }[] = eig.kind === "real" ? [
    { v: eig.vectors[0], lam: eig.values[0], idx: 1 },
    { v: eig.vectors[1], lam: eig.values[1], idx: 2 },
  ] : eig.kind === "defective" ? [{ v: eig.vector, lam: eig.value, idx: 1 }] : [];

  let eigText: ReactNode;
  if (eig.kind === "real") {
    eigText = (
      <>
        λ₁ = <b>{show(eig.values[0])}</b> along ({show(eig.vectors[0][0])}, {show(eig.vectors[0][1])}); λ₂ = <b>{show(eig.values[1])}</b> along ({show(eig.vectors[1][0])}, {show(eig.vectors[1][1])})
      </>
    );
  } else if (eig.kind === "defective") {
    eigText = (
      <>
        λ = <b>{show(eig.value)}</b> (repeated) with only <b>one</b> eigen-direction ({show(eig.vector[0])}, {show(eig.vector[1])}) — the matrix is not diagonalizable.
      </>
    );
  } else if (eig.kind === "scalar") {
    eigText = (
      <>
        A = {show(eig.value)}·I: <b>every</b> direction is an eigenvector with λ = {show(eig.value)}.
      </>
    );
  } else {
    eigText = (
      <>
        <b>Complex eigenvalues</b> λ = {show(eig.re)} ± {show(eig.im)}i: no real direction is left on its own span — the map rotates every line.
      </>
    );
  }

  /* ---- narration ---- */
  const stage = Math.round(pos);
  const moving = Math.abs(pos - s.step) > 1e-3;
  const sv = dec.sigma;
  let title = "";
  let body: ReactNode = null;
  if (mode === "transform") {
    const t = Math.max(0, Math.min(1, pos));
    if (moving) {
      title = `Interpolating M(t) = (1 − t)·I + t·A, t = ${t.toFixed(2)}`;
      body = (
        <>
          Every point moves in a straight line from x to Ax. The determinant passes through {show(detM)}. Points on an eigen-line stay on it the whole way, because M(t)v = (1 − t + tλ)v when Av = λv.
        </>
      );
    } else if (stage === 0) {
      title = "Identity: nothing has moved yet";
      body = (
        <>
          The dashed outline shows where the unit circle will land under A. Press <b>Play</b> or <b>Step</b> to morph the plane from I into A, or drag anywhere on the plane to move the input vector <b>x</b>.
        </>
      );
    } else {
      title = "The plane after A";
      body = (
        <>
          The columns of A are where the basis vectors land: <b className="text-accent">î → ({show(A.a)}, {show(A.c)})</b>, <b className="text-clay">ĵ → ({show(A.b)}, {show(A.d)})</b>. The unit square became a parallelogram of signed area det A = <b>{show(detA)}</b>
          {Math.abs(detA) < 1e-9 ? " — the plane collapsed onto a line (or point): A is singular and cannot be undone." : detA < 0 ? " — negative, so orientation flipped (the square is mirrored)." : "."} {eigText}
        </>
      );
    }
  } else {
    const thV = deg(dec.thetaV);
    const thU = deg(dec.thetaU);
    if (stage === 0 && !moving) {
      title = "SVD: A = U Σ Vᵀ, applied right to left";
      body = (
        <>
          The arrows are the right singular vectors v₁, v₂ — perpendicular unit vectors. Step 1 applies Vᵀ, step 2 applies Σ, step 3 applies U. Their product is exactly A.
        </>
      );
    } else if (pos <= 1) {
      title = `Step 1 · Vᵀ rotates by ${show(-thV, 1)}°`;
      body = (
        <>
          Vᵀ is a rotation, so lengths and the circle are unchanged. It turns v₁ onto the x-axis and v₂ onto the y-axis. det = {show(detM)}.
        </>
      );
    } else if (pos <= 2) {
      title = `Step 2 · Σ stretches the axes by σ₁ = ${show(sv[0], 3)}, σ₂ = ${show(sv[1], 3)}`;
      body = (
        <>
          A diagonal matrix scales each axis independently: the circle becomes an axis-aligned ellipse with semi-axes σ₁ and σ₂. Area scales by σ₁σ₂ = {show(sv[0] * sv[1], 3)} = |det A|.
          {sv[1] < 1e-6 ? " σ₂ = 0: one axis is squashed flat — A has rank 1." : ""}
        </>
      );
    } else {
      title = `Step 3 · U ${dec.uSign === -1 ? "reflects and " : ""}rotates by ${show(thU, 1)}°`;
      body = (
        <>
          U carries the ellipse axes onto the left singular vectors u₁, u₂. Now the arrows are Av₁ = σ₁u₁ and Av₂ = σ₂u₂: the longest and shortest stretch of A. Condition number σ₁/σ₂ ={" "}
          {sv[1] < 1e-9 ? "∞" : show(sv[0] / sv[1], 3)}.{dec.uSign === -1 ? " Because det A < 0, U contains a reflection, animated as the second axis passing through zero." : ""}
        </>
      );
    }
  }

  const presetNote = PRESETS.find((p) => p.key === preset)?.note;
  const svdMats = {
    U: dec.U,
    S: { a: sv[0], b: 0, c: 0, d: sv[1] } as Mat2,
    Vt: transpose(dec.V),
  };

  return (
    <SimShell
      title="Linear Transformations, Eigenvectors & SVD"
      subtitle="Apply a 2×2 matrix to the plane; watch areas, invariant directions and the rotate–scale–rotate decomposition."
      controls={
        <Controls
          s={s}
          extra={
            <div className="flex flex-wrap items-center gap-2">
              <div className="flex overflow-hidden rounded-md border border-border text-xs" role="group" aria-label="Animation mode">
                <button className={`px-2.5 py-1 ${mode === "transform" ? "bg-surface-2 font-semibold" : "text-muted"}`} onClick={() => changeMode("transform")} aria-pressed={mode === "transform"}>
                  I → A
                </button>
                <button className={`border-l border-border px-2.5 py-1 ${mode === "svd" ? "bg-surface-2 font-semibold" : "text-muted"}`} onClick={() => changeMode("svd")} aria-pressed={mode === "svd"}>
                  SVD
                </button>
              </div>
              <label className="flex items-center gap-1.5 text-xs text-muted">
                Preset
                <select
                  className="field py-0.5"
                  value={preset}
                  onChange={(e) => {
                    const p = PRESETS.find((q) => q.key === e.target.value);
                    if (p) editA(p.M, p.key);
                  }}
                  aria-label="Matrix preset"
                >
                  {preset === "custom" && <option value="custom">Custom</option>}
                  {PRESETS.map((p) => (
                    <option key={p.key} value={p.key}>
                      {p.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          }
        />
      }
      aside={
        <div className="space-y-3">
          <Narration title={title}>{body}</Narration>
          <div className="space-y-2 rounded-lg border border-border p-3">
            <MatView name="A" M={A} />
            <MatView name={mode === "transform" ? "M(t)" : "current"} M={M} />
            <div className="font-mono text-xs">
              det A = <b>{show(detA, 3)}</b> · tr A = {show(trace(A), 3)} · det (current) = <span style={{ color: areaColor }}>{show(detM, 3)}</span>
            </div>
            <div className="font-mono text-xs">
              x = ({show(x[0])}, {show(x[1])}) → <span className="text-gold font-semibold">({show(Mx[0])}, {show(Mx[1])})</span>
            </div>
          </div>
          {mode === "svd" ? (
            <div className="space-y-1.5 rounded-lg border border-border p-3">
              <div className="text-xs uppercase tracking-wider text-faint">A = U Σ Vᵀ</div>
              <MatView name="U" M={svdMats.U} note={dec.uSign === -1 ? "det −1" : `rot ${show(deg(dec.thetaU), 1)}°`} />
              <MatView name="Σ" M={svdMats.S} />
              <MatView name="Vᵀ" M={svdMats.Vt} note={`rot ${show(-deg(dec.thetaV), 1)}°`} />
            </div>
          ) : (
            <div className="rounded-lg border border-border p-3 text-xs text-muted">
              <div className="mb-1 uppercase tracking-wider text-faint">Eigen-decomposition of A</div>
              {eigText}
            </div>
          )}
        </div>
      }
    >
      <svg
        ref={svgRef}
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        className="mx-auto block w-full h-auto max-w-[460px] cursor-crosshair select-none"
        role="img"
        aria-label={`The plane transformed by the matrix [[${show(M.a)}, ${show(M.b)}], [${show(M.c)}, ${show(M.d)}]]; determinant ${show(detM)}.`}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
      >
        <defs>
          <clipPath id="lt-clip">
            <rect x={0} y={0} width={SIZE} height={SIZE} />
          </clipPath>
        </defs>
        <rect x={0} y={0} width={SIZE} height={SIZE} fill="var(--surface)" />
        <g clipPath="url(#lt-clip)">
          {/* original grid */}
          {Array.from({ length: 2 * R + 1 }, (_, i) => i - R).map((i) => (
            <g key={`og${i}`}>
              <line x1={HALF + i * k} y1={0} x2={HALF + i * k} y2={SIZE} stroke="var(--border)" strokeWidth={i === 0 ? 1.4 : 0.6} />
              <line x1={0} y1={HALF - i * k} x2={SIZE} y2={HALF - i * k} stroke="var(--border)" strokeWidth={i === 0 ? 1.4 : 0.6} />
            </g>
          ))}

          {/* eigen-lines of A (invariant under every M(t) in I → A mode) */}
          {mode === "transform" &&
            eigenLines.map(({ v, idx }) => {
              const p = scr([-3 * R * v[0], -3 * R * v[1]]);
              const q = scr([3 * R * v[0], 3 * R * v[1]]);
              return <line key={`el${idx}`} x1={p[0]} y1={p[1]} x2={q[0]} y2={q[1]} stroke="var(--text-muted)" strokeWidth={1.2} strokeDasharray="2 5" />;
            })}

          {/* transformed grid */}
          {gridLines.map((g, i) => {
            const p = scr(g.p);
            const q = scr(g.q);
            return <line key={`tg${i}`} x1={p[0]} y1={p[1]} x2={q[0]} y2={q[1]} stroke="var(--text-muted)" strokeOpacity={g.axis ? 0.75 : 0.28} strokeWidth={g.axis ? 1.3 : 0.8} />;
          })}

          {/* image of the unit square */}
          <polygon points={square} fill={areaColor} fillOpacity={0.16} stroke={areaColor} strokeWidth={1.2} />

          {/* target outline (where the circle ends up) and the current image of the unit circle */}
          {Math.abs(pos - (total - 1)) > 1e-3 && <path d={circlePath(A)} fill="none" stroke="var(--text-muted)" strokeWidth={1.2} strokeDasharray="6 5" />}
          <path d={circlePath(M)} fill="none" stroke="var(--text)" strokeWidth={1.8} />

          {/* arrows */}
          {mode === "transform" ? (
            <>
              {eigenLines.map(({ v, lam, idx }) => {
                const t = Math.max(0, Math.min(1, pos));
                const f = 1 - t + t * lam; // eigenvalue of M(t) along v
                return <Arrow key={`ea${idx}`} p={O} q={scr([f * v[0], f * v[1]])} color="var(--text)" width={1.6} label={`λ${idx === 1 ? "₁" : "₂"}=${show(lam)}`} />;
              })}
              <Arrow p={O} q={scr(apply(M, [1, 0]))} color="var(--accent)" label="î" />
              <Arrow p={O} q={scr(apply(M, [0, 1]))} color="var(--clay)" label="ĵ" />
            </>
          ) : (
            <>
              <Arrow p={O} q={scr(apply(M, [dec.V.a, dec.V.c]))} color="var(--accent)" label={pos > 2.98 ? "σ₁u₁" : "v₁"} />
              <Arrow p={O} q={scr(apply(M, [dec.V.b, dec.V.d]))} color="var(--clay)" label={pos > 2.98 ? "σ₂u₂" : "v₂"} />
            </>
          )}

          {/* input vector x (dashed) and its image (solid) */}
          <Arrow p={O} q={scr(x)} color="var(--gold)" width={1.5} dashed />
          <Arrow p={O} q={scr(Mx)} color="var(--gold)" width={2.5} label={mode === "transform" ? "Ax" : undefined} />
          <circle cx={scr(x)[0]} cy={scr(x)[1]} r={5} fill="var(--surface)" stroke="var(--gold)" strokeWidth={2} />
        </g>
        <text x={8} y={SIZE - 8} fontSize={11} fill="var(--text-muted)">
          grid: 1 unit = {k.toFixed(0)} px · view ±{R}
        </text>
      </svg>

      <div className="mt-3 grid gap-x-6 gap-y-1.5 sm:grid-cols-2">
        <Slider label="a = A₁₁" value={A.a} min={-4} max={4} step={0.1} onChange={(v) => editA({ ...A, a: v })} fmt={(v) => v.toFixed(2)} />
        <Slider label="b = A₁₂" value={A.b} min={-4} max={4} step={0.1} onChange={(v) => editA({ ...A, b: v })} fmt={(v) => v.toFixed(2)} />
        <Slider label="c = A₂₁" value={A.c} min={-4} max={4} step={0.1} onChange={(v) => editA({ ...A, c: v })} fmt={(v) => v.toFixed(2)} />
        <Slider label="d = A₂₂" value={A.d} min={-4} max={4} step={0.1} onChange={(v) => editA({ ...A, d: v })} fmt={(v) => v.toFixed(2)} />
        <Slider label="x₁ (input)" value={x[0]} min={-3} max={3} step={0.1} onChange={(v) => setX([v, x[1]])} fmt={(v) => v.toFixed(1)} />
        <Slider label="x₂ (input)" value={x[1]} min={-3} max={3} step={0.1} onChange={(v) => setX([x[0], v])} fmt={(v) => v.toFixed(1)} />
      </div>
      {presetNote && <p className="mt-2 text-xs text-muted">{presetNote}</p>}
      <p className="mt-1 text-xs text-faint">
        Click or drag on the plane to move x (dashed gold); the solid gold arrow is its image. Shaded parallelogram = image of the unit square (clay when det &lt; 0).
      </p>
    </SimShell>
  );
}
