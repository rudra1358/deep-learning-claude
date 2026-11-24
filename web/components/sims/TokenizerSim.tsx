"use client";

import { useEffect, useMemo, useState } from "react";
import { Controls, Narration, SimShell, Slider, useStepper } from "./SimShell";
import { EOW, encode, trainBpe, vocabulary } from "@/lib/sim/bpe";

const DEFAULT_CORPUS = `The cat sat on the mat. The cats sat on the mats.
A dog sat on the log; the dogs sat on the logs.
Newer models train faster, and the newest models train fastest.`;

/** Render a token, showing the end-of-word marker in a faint colour. */
function TokenText({ tok }: { tok: string }) {
  if (tok.endsWith(EOW))
    return (
      <>
        {tok.slice(0, -1)}
        <span className="text-faint">{EOW}</span>
      </>
    );
  return <>{tok}</>;
}

function Chip({ tok, highlight, alt, id }: { tok: string; highlight?: boolean; alt?: boolean; id?: number }) {
  const bg = highlight ? "color-mix(in srgb, var(--gold) 28%, var(--surface))" : alt ? "color-mix(in srgb, var(--accent) 14%, var(--surface))" : "var(--surface-2)";
  return (
    <span className="inline-flex flex-col items-center">
      <span className="rounded border px-1 font-mono text-xs leading-5" style={{ background: bg, borderColor: highlight ? "var(--gold)" : "var(--border)" }}>
        <TokenText tok={tok} />
      </span>
      {id !== undefined && <span className="font-mono text-[10px] text-faint">{id}</span>}
    </span>
  );
}

export default function TokenizerSim() {
  const [corpus, setCorpus] = useState(DEFAULT_CORPUS);
  const [numMerges, setNumMerges] = useState(30);
  const [probe, setProbe] = useState("The newest dogs sat on the catalogs");

  const trace = useMemo(() => trainBpe(corpus, numMerges), [corpus, numMerges]);
  const total = trace.states.length; // state k = after k merges, k = 0..learned
  const s = useStepper(total, { baseMs: 1000 });
  const { reset } = s;
  useEffect(() => {
    reset();
  }, [corpus, numMerges, reset]);

  const k = Math.min(s.step, total - 1);
  const state = trace.states[k];
  const justMerged = k > 0 ? state.merges[k - 1] : null;
  const newTok = justMerged ? justMerged.a + justMerged.b : null;
  const tablePairs = (k > 0 ? trace.states[k - 1].pairs : state.pairs).slice(0, 8);
  const vocab = vocabulary(trace, k);

  const charCount = trace.corpusWords.reduce((a, w) => a + Array.from(w).length, 0);
  const tokenCount = trace.corpusWords.reduce((a, w) => a + state.segs[w].length, 0);
  const shownWords = trace.corpusWords.slice(0, 80);

  const encoded = useMemo(() => encode(probe, trace, k), [probe, trace, k]);
  const probeChars = useMemo(() => Array.from(probe).length, [probe]);
  const unkCount = encoded.filter((e) => e.id === 0).length;

  let narrTitle: string;
  let narr: string;
  if (!justMerged) {
    narrTitle = "Step 0 · characters only";
    narr = `Every word is split into characters plus the end-of-word marker "${EOW}". The base vocabulary is the ${trace.alphabet.length} distinct symbols. The table shows how often each adjacent pair occurs, weighted by word frequency. Press Step to merge the most frequent pair.`;
  } else {
    const runner = tablePairs[1];
    narrTitle = `Merge ${k} · "${justMerged.a}" + "${justMerged.b}" → "${newTok}"`;
    narr = `This pair occurred ${justMerged.count} times, more than any other${runner && runner.count === justMerged.count ? ` (tied with "${runner.a}"+"${runner.b}"; ties are broken alphabetically)` : ""}. Every occurrence in the corpus is replaced by the new symbol "${newTok}", which becomes vocabulary entry ${vocab.length - 1}. Pair counts are then recomputed, because merging changes which symbols are adjacent.`;
  }
  if (k === total - 1 && total - 1 < numMerges) narr += ` Training stopped after ${total - 1} merges: no remaining pair occurs at least twice.`;

  return (
    <SimShell
      title="Byte-Pair Encoding, Merge by Merge"
      subtitle="Text is lowercased, NFKC-normalized and stripped of punctuation; “_” marks the end of a word."
      controls={
        <div className="space-y-2">
          <Controls s={s} />
          <Slider label="Merges to learn" value={numMerges} min={1} max={80} step={1} onChange={setNumMerges} fmt={(v) => String(v)} />
        </div>
      }
      aside={
        <div className="space-y-3">
          <Narration title={narrTitle}>{narr}</Narration>
          <div className="grid grid-cols-3 gap-2 text-center font-mono text-xs">
            <div className="rounded border border-border p-1.5">
              <div className="text-base text-text">{vocab.length}</div>
              <div className="text-faint">vocab size</div>
            </div>
            <div className="rounded border border-border p-1.5">
              <div className="text-base text-text">{tokenCount}</div>
              <div className="text-faint">corpus tokens</div>
            </div>
            <div className="rounded border border-border p-1.5">
              <div className="text-base text-text">{tokenCount ? (charCount / tokenCount).toFixed(2) : "—"}</div>
              <div className="text-faint">chars / token</div>
            </div>
          </div>
          <div>
            <div className="mb-1 text-xs text-muted">Merges learned so far (in order)</div>
            <div className="flex flex-wrap gap-1">
              {state.merges.length === 0 && <span className="text-xs text-faint">none yet</span>}
              {state.merges.map((m, j) => (
                <span key={j} className="font-mono text-[11px]" style={{ color: j === k - 1 ? "var(--gold)" : "var(--text-muted)" }}>
                  {j + 1}.{m.a}+{m.b}
                </span>
              ))}
            </div>
          </div>
        </div>
      }
    >
      <label className="block text-xs text-muted" htmlFor="bpe-corpus">
        Training corpus (edit me)
      </label>
      <textarea id="bpe-corpus" className="mt-1 w-full rounded border border-border bg-surface p-2 font-mono text-xs" rows={4} value={corpus} onChange={(e) => setCorpus(e.target.value)} />

      <div className="mt-3 grid gap-4 md:grid-cols-[230px_1fr]">
        <div>
          <div className="mb-1 text-xs text-muted">{k > 0 ? `Pair counts before merge ${k}` : "Pair counts (top 8)"}</div>
          <table className="w-full font-mono text-xs">
            <thead>
              <tr className="text-left text-faint">
                <th className="font-normal">pair</th>
                <th className="text-right font-normal">count</th>
              </tr>
            </thead>
            <tbody>
              {tablePairs.map((p, j) => {
                const chosen = k > 0 && j === 0;
                const maxC = tablePairs[0]?.count || 1;
                return (
                  <tr key={p.a + "\u0000" + p.b} style={chosen ? { background: "color-mix(in srgb, var(--gold) 22%, transparent)" } : undefined}>
                    <td className="py-0.5">
                      <TokenText tok={p.a} /> + <TokenText tok={p.b} />
                    </td>
                    <td className="py-0.5 text-right">
                      <span className="mr-1 inline-block h-2 align-middle" style={{ width: `${(p.count / maxC) * 50}px`, background: chosen ? "var(--gold)" : "var(--accent)" }} />
                      {p.count}
                    </td>
                  </tr>
                );
              })}
              {tablePairs.length === 0 && (
                <tr>
                  <td className="text-faint" colSpan={2}>
                    no pairs left
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="min-w-0">
          <div className="mb-1 text-xs text-muted">
            Corpus segmented with the first {k} merge{k === 1 ? "" : "s"} ({tokenCount} tokens for {charCount} characters){trace.corpusWords.length > shownWords.length ? `, first ${shownWords.length} words shown` : ""}
          </div>
          <div className="flex flex-wrap gap-x-2.5 gap-y-1.5">
            {shownWords.map((w, wi) => (
              <span key={wi} className="inline-flex gap-px">
                {state.segs[w].map((tok, ti) => (
                  <Chip key={ti} tok={tok} alt={ti % 2 === 1} highlight={tok === newTok} />
                ))}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-5 border-t border-border pt-3">
        <label className="block text-xs text-muted" htmlFor="bpe-probe">
          Encode new text with the {k} learned merge{k === 1 ? "" : "s"}
        </label>
        <input id="bpe-probe" className="mt-1 w-full rounded border border-border bg-surface px-2 py-1 font-mono text-xs" value={probe} onChange={(e) => setProbe(e.target.value)} />
        <div className="mt-2 flex flex-wrap gap-x-2.5 gap-y-1.5">
          {encoded.map((e, j) => {
            const prevWord = j > 0 ? encoded[j - 1].word : -1;
            return (
              <span key={j} className={e.word !== prevWord && j > 0 ? "ml-1.5" : ""}>
                <Chip tok={e.token} id={e.id} alt={e.word % 2 === 1} highlight={e.id === 0} />
              </span>
            );
          })}
        </div>
        <div className="mt-2 font-mono text-xs text-muted">
          {probeChars} characters → {encoded.length} tokens
          {unkCount > 0 && <span className="text-clay"> · {unkCount} unknown symbol{unkCount === 1 ? "" : "s"} mapped to &lt;unk&gt; (id 0), highlighted</span>}
        </div>
      </div>
    </SimShell>
  );
}
