"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import { encode, forward, loadModel, verifyModel, type CharModel, type ModelJSON } from "@/lib/sim/transformer";
import {
  argmax,
  beamStep,
  entropy,
  filterLogits,
  logSoftmax,
  mulberry32,
  repetitionPenalty,
  sampleIndex,
  type Beam,
  type BeamCandidate,
  type FilterOptions,
  type Filtered,
} from "@/lib/sim/decoding";

const MODEL_URL = "/models/char-transformer.json";
const DEFAULT_PROMPT = "alice was beginning to get very ";
const MAX_GEN = 24;
const ROWS = 12;

type Mode = "greedy" | "sample" | "beam";

const show = (c: string) => (c === " " ? "␣" : c);
const pct = (p: number) => (p >= 0.0995 ? (p * 100).toFixed(1) : (p * 100).toFixed(2));

/* ------------------------------------------------------------------ lazy runs */

/** State before choosing generated token k+1 (k tokens already generated). */
interface TokState {
  gen: number[];
  z: number[]; // raw logits for the next token
  zPen: number[]; // after the repetition penalty
  f: Filtered; // temperature + truncation applied to zPen
  rawLp: number[]; // log softmax(z) at T = 1: the model's own probabilities
  next: number; // the token this strategy picks
  u: number | null; // uniform number used for sampling
  seqLogp: number; // Σ log p_model over gen
}

interface BeamState {
  beams: (Beam & Partial<BeamCandidate>)[];
  scored: number;
  topLp: number[]; // log p(next | best beam)
}

/** Computes generation states on demand and caches them, so stepping forward costs one forward pass per token. */
class TokenRun {
  private states: TokState[] = [];
  private rand: () => number;
  constructor(
    private model: CharModel,
    private prompt: number[],
    private mode: "greedy" | "sample",
    private opts: FilterOptions,
    private penalty: number,
    seed: number,
  ) {
    this.rand = mulberry32(seed);
  }
  get(k: number): TokState {
    while (this.states.length <= k) {
      const prev = this.states[this.states.length - 1];
      const gen = prev ? [...prev.gen, prev.next] : [];
      const seqLogp = prev ? prev.seqLogp + prev.rawLp[prev.next] : 0;
      const ids = [...this.prompt, ...gen];
      const tr = forward(this.model, ids);
      const z = tr.logits[tr.logits.length - 1];
      const zPen = repetitionPenalty(z, ids, this.penalty);
      const f = filterLogits(zPen, this.opts);
      let next: number;
      let u: number | null = null;
      if (this.mode === "greedy") next = argmax(zPen);
      else {
        u = this.rand();
        next = sampleIndex(f.probs, u);
      }
      this.states.push({ gen, z, zPen, f, rawLp: logSoftmax(z), next, u, seqLogp });
    }
    return this.states[k];
  }
}

class BeamRun {
  private states: BeamState[] = [];
  constructor(
    private model: CharModel,
    private prompt: number[],
    private width: number,
  ) {}
  private lp = (ids: number[]) => {
    const tr = forward(this.model, [...this.prompt, ...ids]);
    return logSoftmax(tr.logits[tr.logits.length - 1]);
  };
  get(k: number): BeamState {
    while (this.states.length <= k) {
      const prev = this.states[this.states.length - 1];
      if (!prev) {
        this.states.push({ beams: [{ ids: [], logp: 0 }], scored: 0, topLp: this.lp([]) });
      } else {
        const st = beamStep(prev.beams, this.width, this.lp);
        this.states.push({ beams: st.beams, scored: st.scored, topLp: this.lp(st.beams[0].ids) });
      }
    }
    return this.states[k];
  }
}

/* ------------------------------------------------------------------ distribution chart */

function DistChart({ f, raw, vocab, chosen, showCum, beamMode }: { f: Filtered; raw: number[]; vocab: string[]; chosen: Set<number>; showCum: boolean; beamMode: boolean }) {
  const ids = f.order.slice(0, ROWS);
  const W = 440;
  const rh = 22;
  const lab = 34;
  const barW = 210;
  const H = ids.length * rh + 4;
  const max = Math.max(f.probsT[ids[0]], f.probs[ids[0]], ...ids.map((i) => raw[i]), 1e-9);
  let cum = 0;
  const cums = f.order.map((i) => (cum += f.probsT[i]));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Next-character probabilities: kept tokens in colour, removed tokens grey">
      {ids.map((id, r) => {
        const y = r * rh + 2;
        const kept = f.keep[id];
        const wT = (f.probsT[id] / max) * barW;
        const wQ = (f.probs[id] / max) * barW;
        const xRaw = lab + (raw[id] / max) * barW;
        const isChosen = chosen.has(id);
        return (
          <g key={id}>
            <text x={lab - 6} y={y + 15} fontSize={13} textAnchor="end" fill={isChosen ? "var(--gold)" : kept ? "var(--text)" : "var(--text-faint)"} fontFamily="var(--font-mono)" fontWeight={isChosen ? 700 : 400}>
              {`'${show(vocab[id])}'`}
            </text>
            {/* p after temperature (outline) */}
            <rect x={lab} y={y + 3} width={Math.max(1, wT)} height={rh - 8} rx={2} fill="none" stroke={kept ? "var(--accent)" : "var(--border-strong)"} strokeWidth={1} />
            {/* renormalised probability actually used (filled) */}
            {kept && <rect x={lab} y={y + 5} width={Math.max(1, wQ)} height={rh - 12} rx={1.5} fill={isChosen ? "var(--gold)" : "var(--accent)"} fillOpacity={0.85} />}
            {/* model probability at T = 1 (tick) */}
            <line x1={xRaw} x2={xRaw} y1={y + 1} y2={y + rh - 3} stroke="var(--text-muted)" strokeWidth={1.5} />
            <text x={lab + barW + 26} y={y + 15} fontSize={11.5} fill={kept ? "var(--text-muted)" : "var(--text-faint)"} fontFamily="var(--font-mono)">
              {beamMode ? `${pct(f.probsT[id])}%  ln p ${Math.log(f.probsT[id]).toFixed(2)}` : kept ? `${pct(f.probsT[id])}% → ${pct(f.probs[id])}%` : `${pct(f.probsT[id])}% cut`}
              {showCum && !beamMode ? `  Σ${(cums[r] * 100).toFixed(0)}` : ""}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function Legend({ beamMode }: { beamMode: boolean }) {
  return (
    <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-muted">
      {!beamMode && (
        <span className="flex items-center gap-1">
          <span className="inline-block h-2.5 w-4 rounded-sm border" style={{ borderColor: "var(--accent)" }} /> p after temperature
        </span>
      )}
      {!beamMode && (
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-4 rounded-sm" style={{ background: "var(--accent)" }} /> rescaled to sum to 1 (sampled from)
        </span>
      )}
      <span className="flex items-center gap-1">
        <span className="inline-block h-3 w-0.5" style={{ background: "var(--text-muted)" }} /> model p at T = 1
      </span>
      <span className="flex items-center gap-1">
        <span className="inline-block h-2 w-4 rounded-sm" style={{ background: "var(--gold)" }} /> {beamMode ? "extended by a surviving beam" : "picked"}
      </span>
    </div>
  );
}

/* ------------------------------------------------------------------ main component */

export default function DecodingSim() {
  const [model, setModel] = useState<CharModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [check, setCheck] = useState<{ maxAbsDiff: number; checked: number } | null>(null);
  const [input, setInput] = useState(DEFAULT_PROMPT);
  const [mode, setMode] = useState<Mode>("sample");
  const [temperature, setTemperature] = useState(1);
  const [topK, setTopK] = useState(0);
  const [topP, setTopP] = useState(0.9);
  const [minP, setMinP] = useState(0);
  const [penalty, setPenalty] = useState(1);
  const [seed, setSeed] = useState(1);
  const [width, setWidth] = useState(3);

  useEffect(() => {
    let alive = true;
    fetch(MODEL_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} while fetching ${MODEL_URL}`);
        return r.json() as Promise<ModelJSON>;
      })
      .then((j) => {
        if (!alive) return;
        const m = loadModel(j);
        setModel(m);
        setCheck(verifyModel(m));
      })
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, []);

  const s = useStepper(MAX_GEN + 1, { baseMs: 900 });

  const inputLower = input.toLowerCase();
  const unknown = model ? encode(model, inputLower).unknown : [];
  const inputError = !model ? null : inputLower.length === 0 ? "Type at least one character." : unknown.length ? `The model does not know these characters: ${unknown.map((c) => `'${c}'`).join(" ")}` : null;
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  useEffect(() => {
    if (model && !inputError) setPrompt(inputLower);
  }, [model, inputLower, inputError]);

  const promptIds = useMemo(() => (model ? encode(model, prompt).ids : []), [model, prompt]);
  const opts: FilterOptions = useMemo(() => ({ temperature, topK, topP, minP }), [temperature, topK, topP, minP]);
  const tokRun = useMemo(() => (model && mode !== "beam" && promptIds.length ? new TokenRun(model, promptIds, mode, opts, penalty, seed) : null), [model, mode, promptIds, opts, penalty, seed]);
  const beamRun = useMemo(() => (model && mode === "beam" && promptIds.length ? new BeamRun(model, promptIds, width) : null), [model, mode, promptIds, width]);
  const greedyRef = useMemo(() => (model && mode === "beam" && promptIds.length ? new BeamRun(model, promptIds, 1) : null), [model, mode, promptIds]);

  // Changing anything that alters the generated sequence restarts it.
  const { reset } = s;
  useEffect(() => reset(), [reset, prompt, mode, opts, penalty, seed, width]);

  const title = "Decoding Strategies (trained model)";
  const subtitle = "A real Transformer gives odds for the next character. Turn them into text with greedy, sampling (temperature / top-k / top-p), or beam search.";
  if (error) {
    return (
      <SimShell title={title} subtitle={subtitle}>
        <p className="text-sm text-clay">Could not load the model ({error}). Check your internet connection and reload the page.</p>
      </SimShell>
    );
  }
  if (!model || !promptIds.length) {
    return (
      <SimShell title={title} subtitle={subtitle}>
        <div className="flex h-40 items-center justify-center text-sm text-muted" role="status">
          Loading the trained model (about 300 KB of weights)…
        </div>
      </SimShell>
    );
  }

  const k = s.step;
  const vocab = model.vocab;
  const decodeIds = (ids: number[]) => ids.map((i) => vocab[i]).join("");

  /* --------------------------- per-mode state */
  let f: Filtered;
  let raw: number[];
  let chosen = new Set<number>();
  let generated = "";
  let seqLogp = 0;
  let body: ReactNode = null;
  let narration: ReactNode = null;
  let narrTitle = "";

  if (mode !== "beam" && tokRun) {
    const st = tokRun.get(k);
    f = st.f;
    raw = st.rawLp.map(Math.exp);
    chosen = new Set([st.next]);
    generated = decodeIds(st.gen);
    seqLogp = st.seqLogp;
    const nextC = `'${show(vocab[st.next])}'`;
    const top = f.order[0];
    narrTitle = `Token ${k + 1}: ${mode === "greedy" ? "greedy pick" : "sample"} ${nextC}`;
    narration =
      mode === "greedy" ? (
        <>
          The model gives <span className="font-mono">{nextC}</span> the highest {penalty !== 1 ? "penalised " : ""}logit (raw score, {fmt(st.zPen[st.next], 2)}), so greedy takes it. Model probability: {pct(raw[st.next])}%. Greedy has no randomness. Temperature, top-k and top-p cannot change which score is biggest, so it ignores them. So far it has written {k} characters with total log-probability {fmt(seqLogp, 3)} nats.
          {penalty !== 1 && ` The repetition penalty θ = ${penalty.toFixed(2)} shrinks the logits of all ${new Set([...promptIds, ...st.gen]).size} characters already in the text.`}
        </>
      ) : (
        <>
          Temperature T = {temperature.toFixed(2)} {temperature < 1 ? "sharpens" : temperature > 1 ? "flattens" : "leaves"} the odds (entropy, a measure of uncertainty: {fmt(entropy(raw), 3)} → {fmt(entropy(f.probsT), 3)} nats). Truncation keeps {f.keptCount} of {vocab.length} characters, holding {pct(f.keptMass)}% of the probability. We divide their probabilities by {fmt(f.keptMass, 4)} so they add up to 1. A random number u = {fmt(st.u ?? 0, 4)} lands in the slot of <span className="font-mono">{nextC}</span> (q = {pct(f.probs[st.next])}%)
          {st.next !== top ? `, not the most likely '${show(vocab[top])}'` : ""}.
        </>
      );
  } else {
    const bs = (beamRun as BeamRun).get(k);
    const nextBs = k < MAX_GEN ? (beamRun as BeamRun).get(k + 1) : null;
    const gr = (greedyRef as BeamRun).get(k);
    raw = bs.topLp.map(Math.exp);
    f = filterLogits(bs.topLp, { temperature: 1, topK: 0, topP: 1, minP: 0 });
    if (nextBs) chosen = new Set(nextBs.beams.filter((b) => b.parent === 0).map((b) => b.token as number));
    generated = decodeIds(bs.beams[0].ids);
    seqLogp = bs.beams[0].logp;
    const g = gr.beams[0];
    narrTitle = k === 0 ? "Beam search: start" : `Step ${k}: keep the best ${bs.beams.length} of ${bs.scored} options`;
    narration =
      k === 0 ? (
        `Beam search keeps the ${width} best partial texts (beams) at once. Each step tries every beam with all ${vocab.length} characters. It scores each option by its total log-probability Σ ln p and keeps the best ${width}. Width 1 is exactly greedy decoding. Press Step.`
      ) : (
        <>
          We scored {bs.scored} options ({k === 1 ? "one beam" : `${Math.min(width, bs.beams.length)} beams`} × {vocab.length} characters). The best beam has Σ ln p = {fmt(bs.beams[0].logp, 3)} (average {fmt(bs.beams[0].logp / k, 3)} per character). Greedy&apos;s text of the same length has {fmt(g.logp, 3)}.{" "}
          {bs.beams[0].logp > g.logp + 1e-9 ? "Beam search found a more likely text than greedy. Greedy locked in a character that looked best right then, but was not best overall." : "So far greedy's path is still the best one found."}
        </>
      );
    body = (
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-0 font-mono text-[11px] tabular-nums">
          <thead>
            <tr className="text-left text-muted">
              <th className="pr-2 font-normal">#</th>
              <th className="pr-2 font-normal">continuation</th>
              <th className="pr-2 text-right font-normal">Σ ln p</th>
              {k > 0 && <th className="text-right font-normal">from</th>}
            </tr>
          </thead>
          <tbody>
            {bs.beams.map((b, i) => (
              <tr key={i} style={{ color: i === 0 ? "var(--text)" : "var(--text-muted)" }}>
                <td className="pr-2">{i + 1}</td>
                <td className="break-all pr-2">{decodeIds(b.ids).replace(/ /g, "␣") || "∅"}</td>
                <td className="pr-2 text-right">{fmt(b.logp, 3)}</td>
                {k > 0 && <td className="text-right">beam {(b.parent ?? 0) + 1}</td>}
              </tr>
            ))}
            {k > 0 && (
              <tr className="text-faint">
                <td className="pr-2">g</td>
                <td className="break-all pr-2">{decodeIds(g.ids).replace(/ /g, "␣")} (greedy)</td>
                <td className="pr-2 text-right">{fmt(g.logp, 3)}</td>
                {k > 0 && <td />}
              </tr>
            )}
          </tbody>
        </table>
      </div>
    );
  }

  /* --------------------------- controls */
  const controls = (
    <div className="flex flex-col gap-2.5">
      <Controls s={s} />
      <label className="flex min-w-0 items-center gap-2 text-xs">
        <span className="shrink-0 text-muted">Prompt</span>
        <input className="field min-w-0 flex-1 font-mono" value={input} maxLength={160} onChange={(e) => setInput(e.target.value)} aria-label="Prompt text" aria-invalid={!!inputError} />
      </label>
      {inputError && (
        <p className="text-xs text-clay">
          {inputError} The vocabulary is: <span className="font-mono">{vocab.map(show).join("")}</span>
        </p>
      )}
      <div className="flex flex-wrap items-center gap-1 text-xs" role="group" aria-label="Decoding strategy">
        <span className="mr-1 text-muted">Strategy</span>
        {(["greedy", "sample", "beam"] as const).map((m) => (
          <button key={m} className={`sim-btn ${mode === m ? "primary" : ""}`} onClick={() => setMode(m)} aria-pressed={mode === m}>
            {m === "greedy" ? "Greedy" : m === "sample" ? "Sampling" : "Beam search"}
          </button>
        ))}
        {mode === "sample" && (
          <button className="sim-btn" onClick={() => setSeed((v) => v + 1)} aria-label="Change sampling seed">
            seed {seed}
          </button>
        )}
      </div>
      {mode === "beam" ? (
        <Slider label="Beam width" value={width} min={1} max={5} step={1} onChange={setWidth} fmt={(v) => String(v)} />
      ) : (
        <div className="grid gap-1.5 sm:grid-cols-2 sm:gap-x-4">
          <Slider label="Temperature T" value={temperature} min={0.1} max={2} step={0.05} onChange={setTemperature} />
          <Slider label="Top-k" value={topK} min={0} max={vocab.length} step={1} onChange={setTopK} fmt={(v) => (v === 0 ? "off" : String(v))} />
          <Slider label="Top-p" value={topP} min={0.05} max={1} step={0.01} onChange={setTopP} fmt={(v) => (v >= 1 ? "off" : v.toFixed(2))} />
          <Slider label="Min-p" value={minP} min={0} max={0.5} step={0.01} onChange={setMinP} fmt={(v) => (v === 0 ? "off" : v.toFixed(2))} />
          <Slider label="Repetition θ" value={penalty} min={1} max={2} step={0.05} onChange={setPenalty} fmt={(v) => (v === 1 ? "off" : v.toFixed(2))} />
        </div>
      )}
      {mode === "greedy" && <p className="text-[11px] text-muted">Greedy always takes the top choice. Temperature and truncation reshape the chart but cannot change its pick. Only the repetition penalty can.</p>}
    </div>
  );

  const aside = (
    <div className="flex flex-col gap-3">
      <Narration title={narrTitle}>{narration}</Narration>
      <div className="rounded-lg border border-border p-2.5 font-mono text-[11px] leading-snug text-muted">
        {mode === "beam" ? (
          <>
            beam width {width} · step {k}/{MAX_GEN}
            <br />
            best Σ ln p = {fmt(seqLogp, 3)}
          </>
        ) : (
          <>
            kept {f.keptCount}/{vocab.length} · mass {pct(f.keptMass)}%
            <br />
            entropy: model {fmt(entropy(raw), 3)} → after T {fmt(entropy(f.probsT), 3)} → sampled {fmt(entropy(f.probs), 3)} nats
            <br />
            Σ ln p_model(generated) = {fmt(seqLogp, 3)}
          </>
        )}
      </div>
      <div className="rounded-lg border border-border p-2.5 text-[11px] leading-snug text-muted">
        <span className="font-medium text-accent">This is a real trained model.</span> It is the same {model.nParams.toLocaleString()}-parameter, character-level Transformer as in the Transformer Explorer ({model.json.config.n_layers} layers, d_model = {model.json.config.d_model}, {model.json.config.context}-character context). It was trained on Chapter I of Lewis Carroll&apos;s <i>Alice&apos;s Adventures in Wonderland</i> (public domain). Each new character is one forward pass in your browser, and every probability is computed live.
        {check && <span className="font-mono"> your browser vs PyTorch max |Δlogit| = {check.maxAbsDiff.toExponential(1)} {check.maxAbsDiff < 1e-3 ? "✓" : "✗"}</span>}
      </div>
    </div>
  );

  return (
    <SimShell title={title} subtitle={subtitle} controls={controls} aside={aside}>
      <p className="break-words font-mono text-sm leading-relaxed" aria-label="Prompt and generated text">
        <span>{prompt}</span>
        <span className="text-accent">{generated}</span>
        {k < MAX_GEN && mode !== "beam" && <span className="text-gold">{vocab[[...chosen][0] ?? 0]}</span>}
        <span className="animate-pulse text-gold">▍</span>
      </p>
      <p className="mb-3 mt-1 text-[11px] text-muted">
        {k} of {MAX_GEN} characters generated{mode !== "beam" && k < MAX_GEN ? ". The gold character is the next pick" : mode === "beam" ? " (best beam shown)" : ""}.
      </p>
      <div className="mb-1 text-xs font-medium">
        {mode === "beam" ? "Odds for the next character after the best beam (T = 1, no truncation)" : `Odds for the next character: top ${ROWS} of ${vocab.length}, most likely first`}
      </div>
      <DistChart f={f} raw={raw} vocab={vocab} chosen={chosen} showCum={mode !== "beam" && topP < 1} beamMode={mode === "beam"} />
      <Legend beamMode={mode === "beam"} />
      {body}
    </SimShell>
  );
}
