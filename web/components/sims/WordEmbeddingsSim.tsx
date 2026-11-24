"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Controls, Narration, SimShell, Slider, fmt, useStepper } from "./SimShell";
import {
  TOY_CORPUS,
  analogy,
  initToy,
  loadModel,
  neighbours,
  toyCos,
  toyStep,
  verifyModel,
  type Hit,
  type ToyState,
  type ToyUpdate,
  type W2VJSON,
  type W2VModel,
} from "@/lib/sim/wordEmbeddings";

const MODEL_URL = "/models/word2vec.json";
const TITLE = "Word Embedding Explorer";
const SUBTITLE = "Real skip-gram vectors trained with negative sampling, plus a tiny model you can train step by step.";

type Tour = { kind: "nb"; word: string; text: string } | { kind: "an"; a: string; b: string; c: string; text: string };
const TOUR: Tour[] = [
  { kind: "nb", word: "king", text: "Nearest neighbours of 'king' by cosine similarity: the other royal titles. Nobody told the model they are related — they occur in the same kinds of sentences (throne, crown, palace)." },
  { kind: "nb", word: "january", text: "Months cluster tightly and are ordered by season: january's nearest neighbours are the other winter months, because they share weather words (snowy, cold) and appear next to each other in 'X comes after Y'." },
  { kind: "nb", word: "happy", text: "A classic caveat: the nearest neighbours of 'happy' include its opposite 'sad'. Antonyms occur in the same contexts, so distributional similarity is relatedness, not synonymy." },
  { kind: "an", a: "man", b: "king", c: "woman", text: "king − man + woman: the query vector b − a + c lands next to 'queen'. The offset man→king (adding 'royalty') is roughly the same vector as woman→queen." },
  { kind: "an", a: "france", b: "paris", c: "italy", text: "paris − france + italy = rome. Capital–country is one of the best-recovered relations in this model (about 79% top-1 over 2,756 questions)." },
  { kind: "an", a: "walk", b: "walked", c: "swim", text: "An irregular past tense often fails: walked − walk + swim does not give 'swam'. The relation 'past tense' is only 39% correct here — open the evaluation table below." },
  { kind: "nb", word: "orange", text: "Polysemy: one vector per word type. 'orange' is both a fruit and a colour in the corpus; its first ten neighbours are fruits, and colours (red, purple) follow just behind — raise k to 15 to see them." },
];

function useModel() {
  const [model, setModel] = useState<W2VModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [check, setCheck] = useState<ReturnType<typeof verifyModel> | null>(null);
  useEffect(() => {
    let alive = true;
    fetch(MODEL_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} while fetching ${MODEL_URL}`);
        return r.json() as Promise<W2VJSON>;
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
  return { model, error, check };
}

export default function WordEmbeddingsSim() {
  const [tab, setTab] = useState<"explore" | "toy">("explore");
  const { model, error, check } = useModel();
  const tour = useStepper(TOUR.length, { baseMs: 4200 });

  return (
    <SimShell
      title={TITLE}
      subtitle={SUBTITLE}
      controls={
        <div className="flex flex-col gap-2">
          <div className="flex flex-wrap gap-2" role="tablist" aria-label="Simulator mode">
            <button role="tab" aria-selected={tab === "explore"} className={`sim-btn ${tab === "explore" ? "primary" : ""}`} onClick={() => setTab("explore")}>
              Explore trained vectors
            </button>
            <button role="tab" aria-selected={tab === "toy"} className={`sim-btn ${tab === "toy" ? "primary" : ""}`} onClick={() => setTab("toy")}>
              Train a toy model
            </button>
          </div>
          {tab === "explore" && <Controls s={tour} />}
        </div>
      }
    >
      {tab === "explore" ? (
        error ? (
          <p className="text-sm text-clay">Could not load the word vectors ({error}). Check your connection and reload the page.</p>
        ) : !model ? (
          <div className="flex h-40 items-center justify-center text-sm text-muted" role="status">
            Loading the trained word vectors (≈380 KB)…
          </div>
        ) : (
          <Explorer model={model} check={check} tourStep={tour.step} />
        )
      ) : (
        <ToyTrainer />
      )}
    </SimShell>
  );
}

// =============================================================================== explore tab
function Bar({ v, color }: { v: number; color: string }) {
  const w = Math.max(0, Math.min(1, v)) * 100;
  return (
    <div className="h-2 w-full rounded bg-surface-2" aria-hidden>
      <div className="h-2 rounded" style={{ width: `${w}%`, background: color }} />
    </div>
  );
}

function HitList({ hits, onPick, color }: { hits: Hit[]; onPick: (w: string) => void; color: string }) {
  return (
    <ol className="space-y-1 text-sm">
      {hits.map((h, i) => (
        <li key={h.word} className="grid grid-cols-[1.5rem_minmax(0,7rem)_1fr_3.5rem] items-center gap-2">
          <span className="font-mono text-xs text-faint">{i + 1}</span>
          <button className="truncate text-left font-mono hover:underline" onClick={() => onPick(h.word)} title={`select ${h.word}`}>
            {h.word}
          </button>
          <Bar v={h.sim} color={color} />
          <span className="text-right font-mono text-xs">{fmt(h.sim, 3)}</span>
        </li>
      ))}
    </ol>
  );
}

function Explorer({ model, check, tourStep }: { model: W2VModel; check: ReturnType<typeof verifyModel> | null; tourStep: number }) {
  const j = model.json;
  const [query, setQuery] = useState("king");
  const [word, setWord] = useState("king");
  const [k, setK] = useState(10);
  const [abc, setAbc] = useState<[string, string, string]>(["man", "king", "woman"]);
  const [proj, setProj] = useState<"tsne" | "pca">("tsne");
  const [zoom, setZoom] = useState(true);
  const [cat, setCat] = useState(-1);
  const [narr, setNarr] = useState<string>(TOUR[0].text);

  // the guided tour drives the selection when its step changes
  useEffect(() => {
    const t = TOUR[tourStep];
    if (t.kind === "nb") {
      setWord(t.word);
      setQuery(t.word);
    } else setAbc([t.a, t.b, t.c]);
    setNarr(t.text);
  }, [tourStep]);

  const q = query.trim().toLowerCase();
  const known = model.index.has(q);
  useEffect(() => {
    if (known) setWord(q);
  }, [known, q]);
  const suggestions = useMemo(() => (q && !known ? model.words.filter((w) => w.startsWith(q)).slice(0, 8) : []), [model, q, known]);
  const nbs = useMemo(() => neighbours(model, word, k), [model, word, k]);
  const an = useMemo(() => analogy(model, abc[0].trim().toLowerCase(), abc[1].trim().toLowerCase(), abc[2].trim().toLowerCase(), 5), [model, abc]);
  const wi = model.index.get(word) ?? 0;
  const pick = useCallback((w: string) => {
    setQuery(w);
    setWord(w);
    setNarr(`Selected '${w}'. Its neighbours are ranked by cosine similarity of the 50-dimensional vectors; dashed lines show where those neighbours land in the 2-D projection.`);
  }, []);

  const count = j.counts[wi];
  const catName = j.cat[wi] >= 0 ? j.catNames[j.cat[wi]] : "function word / other";

  return (
    <div className="space-y-4">
      <p className="text-xs text-muted">
        Real model: {j.words.length.toLocaleString()} words × {j.config.dim} dimensions, skip-gram with {j.config.negatives} negative samples, window {j.config.window}, {j.config.epochs} epochs, trained by{" "}
        <code>scripts/train-word2vec.py</code> on a <b>generated</b> “Toyland English” corpus ({j.corpus.sentences.toLocaleString()} template sentences, {j.corpus.tokens.toLocaleString()} tokens, {(j.corpus.bytes / 1e6).toFixed(1)} MB) — far more regular than real text.{" "}
        {check && (
          <span>
            Browser check against Python: {check.checked} neighbour/analogy results {check.sameWords ? "identical" : "DIFFER"}, max |Δcos| = {check.maxErr.toExponential(1)}.
          </span>
        )}
      </p>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="min-w-0 space-y-2">
          <label className="flex flex-wrap items-center gap-2 text-sm">
            <span className="text-muted">Word</span>
            <input className="min-w-0 flex-1 rounded border border-border bg-surface px-2 py-1 font-mono" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Search a word" spellCheck={false} />
          </label>
          {!known && q && (
            <div className="text-xs text-clay">
              “{q}” is not in the {j.words.length.toLocaleString()}-word vocabulary — a word2vec model has no vector for unseen words (FastText would build one from subwords).
              {suggestions.length > 0 && (
                <span className="ml-1 text-muted">
                  Did you mean:{" "}
                  {suggestions.map((s) => (
                    <button key={s} className="mr-1 font-mono underline" onClick={() => pick(s)}>
                      {s}
                    </button>
                  ))}
                </span>
              )}
            </div>
          )}
          <Slider label="neighbours k" value={k} min={5} max={20} step={1} onChange={setK} fmt={(v) => String(v)} />
          <div className="text-xs text-muted">
            <b className="font-mono text-[color:var(--gold)]">{word}</b> · count {count.toLocaleString()} · category “{catName}”
          </div>
          <HitList hits={nbs} onPick={pick} color="var(--accent)" />
        </div>

        <div className="min-w-0 space-y-2">
          <div className="text-sm font-medium">Analogy: a is to b as c is to ?</div>
          <div className="flex flex-wrap items-center gap-1 text-sm">
            {(["a", "b", "c"] as const).map((lab, i) => (
              <input key={lab} className="w-24 min-w-0 rounded border border-border bg-surface px-2 py-1 font-mono" value={abc[i]} aria-label={`analogy word ${lab}`} spellCheck={false}
                onChange={(e) => setAbc((x) => { const y: [string, string, string] = [...x]; y[i] = e.target.value; return y; })} />
            ))}
          </div>
          <div className="font-mono text-xs text-muted">
            query = v({abc[1]}) − v({abc[0]}) + v({abc[2]}); rank all words by cosine, excluding the three inputs
          </div>
          <div className="flex flex-wrap gap-1">
            {[["man", "king", "woman"], ["france", "paris", "japan"], ["big", "bigger", "cold"], ["germany", "german", "greece"], ["eat", "eating", "sing"], ["walk", "walked", "swim"]].map((x) => (
              <button key={x.join()} className="sim-btn text-xs" onClick={() => setAbc(x as [string, string, string])}>
                {x[1]} − {x[0]} + {x[2]}
              </button>
            ))}
          </div>
          {an.missing.length ? <div className="text-xs text-clay">Not in vocabulary: {an.missing.join(", ")}</div> : <HitList hits={an.hits} onPick={pick} color="var(--clay)" />}
        </div>
      </div>

      <Scatter model={model} word={word} nbs={nbs} analogyHit={an.hits[0]?.idx} proj={proj} zoom={zoom} cat={cat} onPick={pick} />
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <label className="flex items-center gap-1">
          projection
          <select className="rounded border border-border bg-surface px-1 py-0.5" value={proj} onChange={(e) => setProj(e.target.value as "tsne" | "pca")} aria-label="2-D projection method">
            <option value="tsne">t-SNE</option>
            <option value="pca">PCA</option>
          </select>
        </label>
        <label className="flex items-center gap-1">
          <input type="checkbox" checked={zoom} onChange={(e) => setZoom(e.target.checked)} /> zoom to the selected word
        </label>
        <label className="flex min-w-0 items-center gap-1">
          highlight
          <select className="min-w-0 max-w-[12rem] rounded border border-border bg-surface px-1 py-0.5" value={cat} onChange={(e) => setCat(+e.target.value)} aria-label="Highlight a ground-truth category">
            <option value={-1}>(none)</option>
            {j.catNames.map((c, i) => (
              <option key={c} value={i}>
                {c}
              </option>
            ))}
          </select>
        </label>
      </div>

      <Narration title="What you are looking at">{narr}</Narration>

      <details className="text-sm">
        <summary className="cursor-pointer font-medium">How good is it? Analogy accuracy on {j.analogyOverall.questions.toLocaleString()} questions</summary>
        <div className="mt-2 overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-muted">
                <th className="py-1 pr-2">relation</th>
                <th className="pr-2 text-right">questions</th>
                <th className="pr-2 text-right">3CosAdd top-1</th>
                <th className="text-right">“nearest to c” baseline</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {j.analogyEval.map((r) => (
                <tr key={r.relation} className="border-t border-border">
                  <td className="py-0.5 pr-2 font-sans">{r.relation}</td>
                  <td className="pr-2 text-right">{r.questions}</td>
                  <td className="pr-2 text-right">{(r.top1 * 100).toFixed(1)}%</td>
                  <td className="text-right">{(r.baseline * 100).toFixed(1)}%</td>
                </tr>
              ))}
              <tr className="border-t border-border font-semibold">
                <td className="py-0.5 pr-2 font-sans">overall</td>
                <td className="pr-2 text-right">{j.analogyOverall.questions}</td>
                <td className="pr-2 text-right">{(j.analogyOverall.top1 * 100).toFixed(1)}%</td>
                <td className="text-right">{(j.analogyOverall.baseline * 100).toFixed(1)}%</td>
              </tr>
            </tbody>
          </table>
          <p className="mt-1 text-xs text-muted">
            Every ordered pair of word pairs in a relation is one question (a:b :: c:d). The baseline answers with the nearest neighbour of c alone. Top-10 neighbours share the word&apos;s
            ground-truth category {(j.purity * 100).toFixed(1)}% of the time.
          </p>
        </div>
      </details>
    </div>
  );
}

const SW = 640, SH = 420, PAD = 16;

function Scatter({ model, word, nbs, analogyHit, proj, zoom, cat, onPick }: { model: W2VModel; word: string; nbs: Hit[]; analogyHit?: number; proj: "tsne" | "pca"; zoom: boolean; cat: number; onPick: (w: string) => void }) {
  const j = model.json;
  const P = j[proj];
  const wi = model.index.get(word) ?? 0;
  const nbSet = useMemo(() => new Set(nbs.map((h) => h.idx)), [nbs]);

  // view box in projection coordinates
  const box = useMemo(() => {
    if (!zoom) return { x0: -1.05, x1: 1.05, y0: -1.05, y1: 1.05 };
    const ids = [wi, ...nbs.map((h) => h.idx)];
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const i of ids) {
      x0 = Math.min(x0, P[i][0]); x1 = Math.max(x1, P[i][0]);
      y0 = Math.min(y0, P[i][1]); y1 = Math.max(y1, P[i][1]);
    }
    let w = Math.max(x1 - x0, 0.2) * 1.3, h = Math.max(y1 - y0, 0.2) * 1.3;
    const aspect = (SW - 2 * PAD) / (SH - 2 * PAD);
    if (w / h < aspect) w = h * aspect; else h = w / aspect;
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    return { x0: cx - w / 2, x1: cx + w / 2, y0: cy - h / 2, y1: cy + h / 2 };
  }, [zoom, wi, nbs, P]);

  const X = (x: number) => PAD + ((x - box.x0) / (box.x1 - box.x0)) * (SW - 2 * PAD);
  const Y = (y: number) => SH - PAD - ((y - box.y0) / (box.y1 - box.y0)) * (SH - 2 * PAD);
  const inView = (i: number) => P[i][0] >= box.x0 && P[i][0] <= box.x1 && P[i][1] >= box.y0 && P[i][1] <= box.y1;

  // labels: selected word, neighbours, analogy answer, then frequent words of each category (greedy, no overlaps)
  const labels = useMemo(() => {
    const order: number[] = [wi, ...nbs.map((h) => h.idx)];
    if (analogyHit !== undefined) order.push(analogyHit);
    const perCat = new Map<number, number>();
    for (let i = 0; i < j.words.length; i++) {
      const c = j.cat[i];
      if (c < 0) continue;
      const n = perCat.get(c) ?? 0;
      if (n < (zoom ? 40 : 2) || c === cat) {
        order.push(i);
        perCat.set(c, n + 1);
      }
    }
    const placed: { x: number; y: number; w: number; i: number }[] = [];
    const seen = new Set<number>();
    for (const i of order) {
      if (seen.has(i) || !inView(i)) continue;
      seen.add(i);
      const x = X(P[i][0]) + 4, y = Y(P[i][1]) - 4, w = j.words[i].length * 6.2;
      if (placed.some((p) => x < p.x + p.w && p.x < x + w && Math.abs(p.y - y) < 11)) continue;
      placed.push({ x, y, w, i });
      if (placed.length > 70) break;
    }
    return placed;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wi, nbs, analogyHit, box, proj, cat, zoom]);

  return (
    <svg viewBox={`0 0 ${SW} ${SH}`} className="h-auto w-full rounded-lg border border-border bg-surface" role="img" aria-label={`${proj === "tsne" ? "t-SNE" : "PCA"} projection of ${j.words.length} word vectors, centred on ${word}`}>
      {nbs.map((h) => (
        <line key={`l${h.idx}`} x1={X(P[wi][0])} y1={Y(P[wi][1])} x2={X(P[h.idx][0])} y2={Y(P[h.idx][1])} stroke="var(--accent)" strokeOpacity={0.45} strokeDasharray="3 3" />
      ))}
      {P.map((p, i) => {
        if (!inView(i)) return null;
        const isSel = i === wi, isNb = nbSet.has(i), isCat = cat >= 0 && j.cat[i] === cat, isAn = i === analogyHit;
        const fill = isSel ? "var(--gold)" : isNb ? "var(--accent)" : isAn ? "var(--clay)" : isCat ? "var(--gold)" : "var(--text-muted)";
        return (
          <circle key={i} cx={X(p[0])} cy={Y(p[1])} r={isSel ? 6 : isNb || isAn ? 4 : isCat ? 3.2 : 2.2} fill={fill} fillOpacity={isSel || isNb || isAn || isCat ? 0.95 : 0.3}
            onClick={() => onPick(j.words[i])} style={{ cursor: "pointer" }}>
            <title>{j.words[i]}</title>
          </circle>
        );
      })}
      {labels.map((l) => (
        <text key={`t${l.i}`} x={l.x} y={l.y} fontSize={11} fill={l.i === wi ? "var(--text)" : nbSet.has(l.i) || l.i === analogyHit ? "var(--text)" : "var(--text-muted)"} fontWeight={l.i === wi ? 700 : 400} style={{ pointerEvents: "none" }}>
          {j.words[l.i]}
        </text>
      ))}
      <text x={SW - PAD} y={SH - 4} fontSize={10} textAnchor="end" fill="var(--text-muted)">
        {proj === "tsne" ? "t-SNE (cosine)" : "PCA"} of the 50-D vectors · gold = selected · accent = its {nbs.length} nearest neighbours in 50-D · clay = analogy answer
      </text>
    </svg>
  );
}

// =============================================================================== toy trainer
const TW = 380, TH = 340;

function ToyTrainer() {
  const [lr, setLr] = useState(0.1);
  const [negK, setNegK] = useState(3);
  const [win, setWin] = useState(2);
  const [seed, setSeed] = useState(1);
  const [showU, setShowU] = useState(false);
  const stRef = useRef<ToyState>(initToy({ window: 2, negatives: 3, lr: 0.1, seed: 1 }));
  const [upd, setUpd] = useState<ToyUpdate | null>(null);
  const [, setVersion] = useState(0);

  const tick = useCallback(() => {
    setUpd(toyStep(stRef.current));
    setVersion((v) => v + 1);
  }, []);
  const s = useStepper(Infinity, { baseMs: 900, onTick: tick });

  const reinit = useCallback(() => {
    stRef.current = initToy({ window: win, negatives: negK, lr, seed });
    setUpd(null);
    setVersion((v) => v + 1);
  }, [win, negK, lr, seed]);
  // structural settings (window, K, seed) restart training; the learning rate applies immediately
  useEffect(() => {
    reinit();
    s.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [win, negK, seed]);
  useEffect(() => {
    stRef.current.cfg.lr = lr;
  }, [lr]);
  const stepper = { ...s, reset: () => { s.reset(); reinit(); } };

  const fastForward = (epochs: number) => {
    const st = stRef.current;
    let last: ToyUpdate | null = null;
    for (let i = 0; i < epochs * st.pairs.length; i++) last = toyStep(st);
    setUpd(last);
    setVersion((v) => v + 1);
  };

  const st = stRef.current;
  const W = st.words;
  const all = [...st.v, ...(showU ? st.u : [])];
  const lim = Math.max(1.5, Math.ceil(Math.max(...all.flat().map(Math.abs)) * 2) / 2 + 0.25);
  const X = (x: number) => TW / 2 + (x / lim) * (TW / 2 - 14);
  const Y = (y: number) => TH / 2 - (y / lim) * (TH / 2 - 14);
  const MAG = 5;
  const role = (i: number) => (upd ? (i === upd.c ? "c" : i === upd.o ? "o" : upd.negs.includes(i) ? "n" : "") : "");
  const col = (r: string) => (r === "c" ? "var(--gold)" : r === "o" ? "var(--accent)" : r === "n" ? "var(--clay)" : "var(--text-muted)");
  const id = (w: string) => W.indexOf(w);
  const pairsCos: [string, string][] = [["king", "queen"], ["man", "woman"], ["cat", "dog"], ["milk", "water"], ["king", "milk"]];

  const arrow = (from: number[], to: number[], color: string, key: string) => {
    const x1 = X(from[0]), y1 = Y(from[1]);
    const x2 = X(from[0] + (to[0] - from[0]) * MAG), y2 = Y(from[1] + (to[1] - from[1]) * MAG);
    return <line key={key} x1={x1} y1={y1} x2={x2} y2={y2} stroke={`var(--${color})`} strokeWidth={2} markerEnd={`url(#we-arrow-${color})`} />;
  };

  const hist = st.lossHist;
  const hmax = Math.max(...hist, 1), hmin = Math.min(...hist, 0);

  return (
    <div className="space-y-3">
      <Controls
        s={stepper}
        canPrev={false}
        extra={
          <span className="flex flex-wrap gap-1">
            <button className="sim-btn text-xs" onClick={() => fastForward(1)}>+1 epoch</button>
            <button className="sim-btn text-xs" onClick={() => fastForward(20)}>+20 epochs</button>
          </span>
        }
      />
      <div className="grid gap-2 sm:grid-cols-2">
        <Slider label="learning rate η" value={lr} min={0.01} max={0.5} step={0.01} onChange={setLr} />
        <Slider label="negatives K" value={negK} min={1} max={6} step={1} onChange={setNegK} fmt={(v) => String(v)} />
        <Slider label="window" value={win} min={1} max={2} step={1} onChange={setWin} fmt={(v) => String(v)} />
        <Slider label="seed" value={seed} min={1} max={9} step={1} onChange={setSeed} fmt={(v) => String(v)} />
      </div>
      <p className="text-xs text-muted">
        Corpus ({TOY_CORPUS.length} sentences, stop words removed): {TOY_CORPUS.map((x) => `“${x}”`).join(" ")}. {W.length} word types, {st.pairs.length} (centre, context) pairs per epoch, d = 2 so the vectors can be drawn directly. Both tables start random (word2vec itself starts the output table at zero).
      </p>
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          <svg viewBox={`0 0 ${TW} ${TH}`} className="h-auto w-full rounded-lg border border-border bg-surface" role="img" aria-label="Two-dimensional input vectors of the toy skip-gram model">
            <defs>
              {["gold", "accent", "clay"].map((c) => (
                <marker key={c} id={`we-arrow-${c}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                  <path d="M0,0 L10,5 L0,10 z" fill={`var(--${c})`} />
                </marker>
              ))}
            </defs>
            <line x1={0} y1={TH / 2} x2={TW} y2={TH / 2} stroke="var(--border)" />
            <line x1={TW / 2} y1={0} x2={TW / 2} y2={TH} stroke="var(--border)" />
            {showU && st.u.map((p, i) => (
              <g key={`u${i}`}>
                <circle cx={X(p[0])} cy={Y(p[1])} r={4} fill="none" stroke={col(role(i) === "c" ? "" : role(i))} strokeWidth={1.5} />
                <text x={X(p[0]) + 5} y={Y(p[1]) + 12} fontSize={9} fill="var(--text-muted)">u:{W[i]}</text>
              </g>
            ))}
            {st.v.map((p, i) => {
              const r = role(i);
              return (
                <g key={`v${i}`}>
                  <circle cx={X(p[0])} cy={Y(p[1])} r={r === "c" ? 6 : 4} fill={r === "c" ? "var(--gold)" : "var(--text-muted)"} fillOpacity={r === "c" ? 1 : 0.6} />
                  <text x={X(p[0]) + 6} y={Y(p[1]) - 5} fontSize={11} fill={r === "c" ? "var(--text)" : "var(--text-muted)"} fontWeight={r === "c" ? 700 : 400}>{W[i]}</text>
                </g>
              );
            })}
            {upd && arrow(upd.vBefore, upd.vAfter, "gold", "av")}
            {upd && showU && arrow(upd.uoBefore, upd.uoAfter, "accent", "ao")}
            {upd && showU && upd.negs.map((n, k) => arrow(upd.unBefore[k], upd.unAfter[k], "clay", `an${k}`))}
            <text x={6} y={TH - 6} fontSize={9} fill="var(--text-muted)">filled = input vectors v · {showU ? "rings = output vectors u · " : ""}update arrows ×{MAG}</text>
          </svg>
          <div className="mt-1 flex flex-wrap items-center gap-3 text-xs">
            <label className="flex items-center gap-1">
              <input type="checkbox" checked={showU} onChange={(e) => setShowU(e.target.checked)} /> show output vectors u
            </label>
            <span className="text-muted">epoch {st.epoch} · updates {st.updates}</span>
          </div>
        </div>
        <div className="min-w-0 space-y-2 text-xs">
          {upd ? <UpdateTable u={upd} words={W} lr={st.cfg.lr} /> : <p className="text-muted">Press <b>Step</b> to perform one SGD update on one (centre, context) pair.</p>}
          <div>
            <div className="mb-1 font-medium">Mean loss per epoch</div>
            {hist.length ? (
              <svg viewBox="0 0 300 70" className="h-auto w-full" role="img" aria-label="Mean training loss per epoch">
                <polyline fill="none" stroke="var(--accent)" strokeWidth={1.5}
                  points={hist.map((h, i) => `${4 + (i / Math.max(1, hist.length - 1)) * 292},${64 - ((h - hmin) / (hmax - hmin || 1)) * 58}`).join(" ")} />
                <text x={4} y={10} fontSize={9} fill="var(--text-muted)">{fmt(hist[0], 3)}</text>
                <text x={296} y={68} fontSize={9} textAnchor="end" fill="var(--text-muted)">epoch {hist.length}: {fmt(hist[hist.length - 1], 3)}</text>
              </svg>
            ) : (
              <p className="text-muted">appears after the first full epoch</p>
            )}
          </div>
          <div className="grid grid-cols-2 gap-x-3 font-mono">
            {pairsCos.map(([a, b]) => (
              <div key={a + b}>cos({a}, {b}) = {fmt(toyCos(st, id(a), id(b)), 2)}</div>
            ))}
          </div>
        </div>
      </div>
      <Narration title={upd ? `Update ${st.updates}: centre “${W[upd.c]}”, context “${W[upd.o]}”` : "Skip-gram with negative sampling, one pair at a time"}>
        {upd ? (
          <>
            The true pair scores s₊ = u<sub>{W[upd.o]}</sub>·v<sub>{W[upd.c]}</sub> = {fmt(upd.sPos, 3)}, so the model believes it with probability σ(s₊) = {fmt(upd.sigPos, 3)}. The {upd.negs.length} noise words ({upd.negs.map((n) => W[n]).join(", ")}) were drawn from
            the unigram distribution raised to 0.75. The loss −log σ(s₊) − Σ log σ(−sₖ) = {fmt(upd.loss, 3)}. The gradient pulls u<sub>{W[upd.o]}</sub> towards v<sub>{W[upd.c]}</sub> (scale 1 − σ(s₊) = {fmt(1 - upd.sigPos, 3)}),
            pushes each noise vector away (scale σ(sₖ)), and moves v<sub>{W[upd.c]}</sub> towards u<sub>{W[upd.o]}</sub> and away from the noise vectors. Train for a few dozen epochs: words that share contexts (king/queen, man/woman, milk/water) end up pointing in the same direction.
          </>
        ) : (
          <>Each update takes one centre word and one word from its window, plus K random “noise” words, and nudges three kinds of vectors with a logistic-regression gradient. Words never compared directly still end up close, because they are pulled towards the same context vectors.</>
        )}
      </Narration>
    </div>
  );
}

function V2({ v }: { v: number[] }) {
  return <span className="font-mono">({fmt(v[0], 3)}, {fmt(v[1], 3)})</span>;
}

function UpdateTable({ u, words, lr }: { u: ToyUpdate; words: string[]; lr: number }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="text-left text-muted">
            <th className="pr-2">vector</th>
            <th className="pr-2">score s</th>
            <th className="pr-2">σ(s)</th>
            <th className="pr-2">gradient</th>
            <th>new value</th>
          </tr>
        </thead>
        <tbody className="font-mono">
          <tr className="border-t border-border">
            <td className="pr-2" style={{ color: "var(--gold)" }}>v {words[u.c]}</td>
            <td className="pr-2">—</td>
            <td className="pr-2">—</td>
            <td className="pr-2"><V2 v={u.gV} /></td>
            <td><V2 v={u.vAfter} /></td>
          </tr>
          <tr className="border-t border-border">
            <td className="pr-2" style={{ color: "var(--accent)" }}>u {words[u.o]} (+)</td>
            <td className="pr-2">{fmt(u.sPos, 3)}</td>
            <td className="pr-2">{fmt(u.sigPos, 3)}</td>
            <td className="pr-2"><V2 v={u.gUo} /></td>
            <td><V2 v={u.uoAfter} /></td>
          </tr>
          {u.negs.map((n, k) => (
            <tr key={k} className="border-t border-border">
              <td className="pr-2" style={{ color: "var(--clay)" }}>u {words[n]} (−)</td>
              <td className="pr-2">{fmt(u.sNeg[k], 3)}</td>
              <td className="pr-2">{fmt(u.sigNeg[k], 3)}</td>
              <td className="pr-2"><V2 v={u.gUn[k]} /></td>
              <td><V2 v={u.unAfter[k]} /></td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-1 text-muted">
        v before <V2 v={u.vBefore} /> · loss {fmt(u.loss, 4)} · η = {fmt(lr, 2)} · every vector moves by −η × gradient
      </div>
    </div>
  );
}
