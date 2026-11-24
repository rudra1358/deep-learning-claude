"use client";

import { useMemo, useState, type ReactNode } from "react";
import { Controls, Narration, SimShell, fmt, seqColor, signedColor, useStepper } from "./SimShell";
import { D_MODEL, MAX_TOKENS, maxAbs, mhaParamCount, runAttention, type AttentionResult } from "@/lib/sim/attention";
import type { Matrix } from "@/lib/sim/random";

const STAGES = [
  "Tokens & IDs",
  "Embeddings + PE",
  "Q, K, V",
  "Scores QKᵀ",
  "Scale 1/√dₖ",
  "Mask",
  "Softmax",
  "Weights · V",
  "Concat & W_O",
] as const;

const HEAD_OPTIONS = [1, 2, 4] as const;

/* ------------------------------------------------------------------ */
/* Matrix heatmap with numbers                                         */
/* ------------------------------------------------------------------ */

type ColorMode = "signed" | "seq";

function cellStyle(v: number, mode: ColorMode, scale: number) {
  if (!Number.isFinite(v)) return { background: "var(--surface-2)", color: "var(--text-faint)" };
  const intensity = mode === "seq" ? Math.max(0, Math.min(1, v)) : 0.15 + 0.85 * Math.min(1, Math.abs(v) / scale);
  return {
    background: mode === "seq" ? seqColor(v) : signedColor(v, scale),
    color: intensity > 0.6 ? "var(--bg)" : "var(--text)",
  };
}

function MatrixView({
  title,
  shape,
  M,
  rowLabels,
  colLabels,
  mode = "signed",
  digits = 2,
  highlightRow,
  onRowClick,
  rowSums = false,
  colGroups,
}: {
  title: ReactNode;
  shape?: string;
  M: Matrix;
  rowLabels: string[];
  colLabels: string[];
  mode?: ColorMode;
  digits?: number;
  highlightRow?: number | null;
  onRowClick?: (i: number) => void;
  rowSums?: boolean;
  /** Optional column-group size: draws a thicker divider every `colGroups` columns (head boundaries). */
  colGroups?: number;
}) {
  const scale = mode === "seq" ? 1 : maxAbs(M);
  return (
    <figure className="min-w-0">
      <figcaption className="mb-1 flex flex-wrap items-baseline gap-x-2 text-xs">
        <span className="font-medium text-text">{title}</span>
        {shape && <span className="font-mono text-faint">{shape}</span>}
      </figcaption>
      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-[2px] font-mono text-[10.5px] leading-none">
          <thead>
            <tr>
              <th />
              {colLabels.map((c, j) => (
                <th
                  key={j}
                  className="px-1 pb-1 text-center font-normal text-muted"
                  style={colGroups && j > 0 && j % colGroups === 0 ? { borderLeft: "2px solid var(--gold)" } : undefined}
                >
                  {c}
                </th>
              ))}
              {rowSums && <th className="px-1 pb-1 text-center font-normal text-gold">Σ row</th>}
            </tr>
          </thead>
          <tbody>
            {M.map((row, i) => {
              const hl = highlightRow === i;
              return (
                <tr
                  key={i}
                  onClick={onRowClick ? () => onRowClick(i) : undefined}
                  className={onRowClick ? "cursor-pointer" : undefined}
                  aria-label={onRowClick ? `Select query ${rowLabels[i]}` : undefined}
                >
                  <th
                    scope="row"
                    className="max-w-[5.5rem] truncate pr-1.5 text-right font-normal"
                    style={{ color: hl ? "var(--gold)" : "var(--text-muted)", fontWeight: hl ? 600 : 400 }}
                  >
                    {rowLabels[i]}
                  </th>
                  {row.map((v, j) => (
                    <td
                      key={j}
                      className="min-w-[2.6rem] rounded-[3px] px-1 py-[5px] text-center tabular-nums"
                      style={{
                        ...cellStyle(v, mode, scale),
                        outline: hl ? "1.5px solid var(--gold)" : undefined,
                        borderLeft: colGroups && j > 0 && j % colGroups === 0 ? "2px solid var(--gold)" : undefined,
                      }}
                      title={Number.isFinite(v) ? v.toFixed(5) : "−∞ (masked)"}
                    >
                      {Number.isFinite(v) ? v.toFixed(digits) : "−∞"}
                    </td>
                  ))}
                  {rowSums && (
                    <td className="px-1 text-center tabular-nums text-gold">{row.reduce((a, b) => a + b, 0).toFixed(3)}</td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </figure>
  );
}

/* ------------------------------------------------------------------ */
/* Arc / line diagram: selected query → every key                      */
/* ------------------------------------------------------------------ */

function ConnectionDiagram({ tokens, weights, masked, q, onPick }: { tokens: string[]; weights: Matrix; masked: Matrix; q: number; onPick: (i: number) => void }) {
  const n = tokens.length;
  const W = 640;
  const H = 190;
  const pad = 40;
  const x = (i: number) => (n === 1 ? W / 2 : pad + (i * (W - 2 * pad)) / (n - 1));
  const yQ = 34;
  const yK = 150;
  const row = weights[q] ?? [];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`Attention weights from query "${tokens[q]}" to every key; line thickness is proportional to the weight.`}>
      <text x={6} y={yQ + 4} fontSize={10} fill="var(--text-faint)">query</text>
      <text x={6} y={yK + 4} fontSize={10} fill="var(--text-faint)">key</text>
      {row.map((w, j) => {
        const isMasked = !Number.isFinite(masked[q]?.[j] ?? 0);
        return (
          <line
            key={j}
            x1={x(q)}
            y1={yQ + 10}
            x2={x(j)}
            y2={yK - 12}
            stroke={isMasked ? "var(--border-strong)" : "var(--accent)"}
            strokeWidth={isMasked ? 1 : 0.6 + 12 * w}
            strokeDasharray={isMasked ? "3 4" : undefined}
            strokeOpacity={isMasked ? 0.8 : 0.35 + 0.65 * w}
            strokeLinecap="round"
          />
        );
      })}
      {tokens.map((t, i) => (
        <g key={`q${i}`} onClick={() => onPick(i)} className="cursor-pointer" role="button" aria-label={`Select query ${t}`}>
          <rect x={x(i) - 28} y={yQ - 12} width={56} height={22} rx={6} fill={i === q ? "var(--gold-soft)" : "var(--surface)"} stroke={i === q ? "var(--gold)" : "var(--border-strong)"} />
          <text x={x(i)} y={yQ + 3} fontSize={11} textAnchor="middle" fill="var(--text)" fontWeight={i === q ? 600 : 400}>
            {t.length > 7 ? t.slice(0, 6) + "…" : t}
          </text>
        </g>
      ))}
      {tokens.map((t, j) => (
        <g key={`k${j}`}>
          <rect x={x(j) - 28} y={yK - 12} width={56} height={22} rx={6} fill="var(--surface)" stroke="var(--border-strong)" />
          <text x={x(j)} y={yK + 3} fontSize={11} textAnchor="middle" fill="var(--text)">
            {t.length > 7 ? t.slice(0, 6) + "…" : t}
          </text>
          <text x={x(j)} y={yK + 26} fontSize={10} textAnchor="middle" fill="var(--text-muted)" fontFamily="var(--font-mono)">
            {(row[j] ?? 0).toFixed(3)}
          </text>
        </g>
      ))}
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/* Per-query numeric breakdown: score → scale → mask → exp → softmax   */
/* ------------------------------------------------------------------ */

function QueryBreakdown({ r, h, q }: { r: AttentionResult; h: number; q: number }) {
  const hr = r.heads[h];
  const raw = hr.scores[q] ?? [];
  const scaled = hr.scaled[q] ?? [];
  const masked = hr.masked[q] ?? [];
  const finite = masked.filter(Number.isFinite);
  const m = finite.length ? Math.max(...finite) : 0;
  const ex = masked.map((v) => (Number.isFinite(v) ? Math.exp(v - m) : 0));
  const Z = ex.reduce((a, b) => a + b, 0);
  return (
    <div className="mt-3">
      <div className="mb-1 text-xs font-medium">
        Query <span className="text-gold">“{r.tokens[q]}”</span> (row {q}), head {h + 1}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full font-mono text-[10.5px] tabular-nums">
          <thead className="text-muted">
            <tr>
              <th className="text-left font-normal">key</th>
              <th className="text-right font-normal">q·k</th>
              <th className="text-right font-normal">÷√{r.dK}</th>
              <th className="text-right font-normal">e^(s−max)</th>
              <th className="text-right font-normal">α</th>
            </tr>
          </thead>
          <tbody>
            {r.tokens.map((t, j) => (
              <tr key={j} className={Number.isFinite(masked[j]) ? "" : "text-faint"}>
                <td className="max-w-[4rem] truncate">{t}</td>
                <td className="text-right">{fmt(raw[j], 3)}</td>
                <td className="text-right">{Number.isFinite(masked[j]) ? fmt(scaled[j], 3) : "−∞"}</td>
                <td className="text-right">{fmt(ex[j], 3)}</td>
                <td className="text-right text-accent">{fmt(hr.weights[q][j], 3)}</td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr className="border-t border-border">
              <td colSpan={3} className="pt-1 text-muted">Σ (max = {fmt(m, 3)})</td>
              <td className="pt-1 text-right">{fmt(Z, 3)}</td>
              <td className="pt-1 text-right text-gold">{fmt(hr.weights[q].reduce((a, b) => a + b, 0), 3)}</td>
            </tr>
          </tfoot>
        </table>
      </div>
      <p className="mt-1 text-[11px] leading-snug text-muted">
        α<sub>j</sub> = e<sup>s<sub>j</sub>−max</sup> / Σ. Subtracting the row maximum does not change the result but prevents overflow.
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Main simulator                                                      */
/* ------------------------------------------------------------------ */

export default function AttentionSim() {
  const [sentence, setSentence] = useState("the cat sat on the mat");
  const [heads, setHeads] = useState<1 | 2 | 4>(2);
  const [usePE, setUsePE] = useState(true);
  const [causal, setCausal] = useState(false);
  const [seed, setSeed] = useState(1);
  const [headSel, setHeadSel] = useState(0);
  const [querySel, setQuerySel] = useState(0);
  const s = useStepper(STAGES.length, { baseMs: 2600 });

  const r = useMemo(() => runAttention({ sentence, heads, usePE, causal, seed }), [sentence, heads, usePE, causal, seed]);
  const n = r.tokens.length;
  const h = Math.min(headSel, heads - 1);
  const q = Math.min(querySel, Math.max(0, n - 1));
  const hr = r.heads[h];
  const dK = r.dK;
  const stage = s.step;

  const tokLabels = r.tokens.map((t, i) => `${i}:${t}`);
  const dimLabels = (k: number, prefix = "d") => Array.from({ length: k }, (_, j) => `${prefix}${j}`);

  const controls = (
    <div className="flex flex-col gap-2.5">
      <Controls s={s} />
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
        <label className="flex min-w-0 flex-1 basis-60 items-center gap-2">
          <span className="shrink-0 text-muted">Sentence</span>
          <input
            className="field min-w-0 flex-1 font-mono"
            value={sentence}
            maxLength={120}
            onChange={(e) => setSentence(e.target.value)}
            aria-label="Input sentence"
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="text-muted">Heads</span>
          <select className="field" value={heads} onChange={(e) => setHeads(+e.target.value as 1 | 2 | 4)} aria-label="Number of heads">
            {HEAD_OPTIONS.map((k) => (
              <option key={k} value={k}>
                {k} (dₖ = {D_MODEL / k})
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={usePE} onChange={(e) => setUsePE(e.target.checked)} /> Positional encoding
        </label>
        <label className="flex items-center gap-1.5">
          <input type="checkbox" checked={causal} onChange={(e) => setCausal(e.target.checked)} /> Causal mask
        </label>
        <button className="sim-btn" onClick={() => setSeed((v) => v + 1)} aria-label="Draw new random projection weights">
          New weights (seed {seed})
        </button>
      </div>
      {heads > 1 && (
        <div className="flex flex-wrap items-center gap-1.5 text-xs" role="group" aria-label="Select head">
          <span className="text-muted">Head</span>
          {Array.from({ length: heads }, (_, k) => (
            <button key={k} className={`sim-btn ${k === h ? "primary" : ""}`} onClick={() => setHeadSel(k)} aria-pressed={k === h}>
              {k + 1}
            </button>
          ))}
        </div>
      )}
    </div>
  );

  const narration = stageNarration(stage, r, h, q, { heads, usePE, causal });

  const aside = (
    <div className="flex flex-col gap-3">
      <Narration title={`${stage + 1}. ${STAGES[stage]}`}>{narration}</Narration>
      {n > 0 && stage >= 3 && <QueryBreakdown r={r} h={h} q={q} />}
      <div className="rounded-lg border border-border p-2.5 text-[11px] leading-snug text-muted">
        <span className="font-medium text-clay">Untrained weights.</span> E, W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> and W<sub>O</sub> are seeded random numbers, so the attention
        pattern carries no linguistic meaning. What is exact is the computation: every number shown is the number computed.
      </div>
    </div>
  );

  let body: ReactNode = null;
  if (n === 0) {
    body = <p className="text-sm text-muted">Type a sentence to begin.</p>;
  } else if (stage === 0) {
    body = (
      <div>
        <p className="mb-3 text-xs text-muted">
          Whitespace tokenization, lower-cased, max {MAX_TOKENS} tokens. IDs come from a toy hashed vocabulary of 10,000 entries, so a repeated word gets the same ID.
          Click a token to make it the query.
        </p>
        <div className="flex flex-wrap gap-2">
          {r.tokens.map((t, i) => (
            <button
              key={i}
              onClick={() => setQuerySel(i)}
              className="rounded-lg border px-3 py-2 text-left"
              style={{ borderColor: i === q ? "var(--gold)" : "var(--border-strong)", background: i === q ? "var(--gold-soft)" : "var(--surface)" }}
              aria-pressed={i === q}
            >
              <div className="font-mono text-sm">{t}</div>
              <div className="font-mono text-[10px] text-muted">pos {i} · id {r.ids[i]}</div>
            </button>
          ))}
        </div>
        <p className="mt-3 font-mono text-xs text-muted">n = {n} tokens → ids [{r.ids.join(", ")}]</p>
      </div>
    );
  } else if (stage === 1) {
    body = (
      <div className="grid gap-4">
        <MatrixView title="Token embeddings E" shape={`${n} × ${D_MODEL}`} M={r.E} rowLabels={tokLabels} colLabels={dimLabels(D_MODEL)} highlightRow={q} onRowClick={setQuerySel} />
        {usePE ? (
          <>
            <MatrixView title="Sinusoidal positional encoding PE" shape={`${n} × ${D_MODEL}`} M={r.PE} rowLabels={tokLabels} colLabels={dimLabels(D_MODEL)} highlightRow={q} onRowClick={setQuerySel} />
            <MatrixView title="Input X = E + PE" shape={`${n} × ${D_MODEL}`} M={r.X} rowLabels={tokLabels} colLabels={dimLabels(D_MODEL)} highlightRow={q} onRowClick={setQuerySel} />
          </>
        ) : (
          <p className="text-xs text-clay">Positional encoding is off: X = E. Repeated words now have identical rows, and attention cannot tell positions apart.</p>
        )}
      </div>
    );
  } else if (stage === 2) {
    const a = h * dK;
    body = (
      <div className="grid gap-4">
        <p className="text-xs text-muted">
          Full projections Q = X·W<sub>Q</sub>, K = X·W<sub>K</sub>, V = X·W<sub>V</sub> are each {n} × {D_MODEL}. Head {h + 1} uses columns {a}…{a + dK - 1}, i.e. {n} × {dK} slices.
        </p>
        <div className="grid gap-4 xl:grid-cols-3">
          <MatrixView title={`Q (head ${h + 1})`} shape={`${n} × ${dK}`} M={hr.Q} rowLabels={tokLabels} colLabels={dimLabels(dK, "q")} highlightRow={q} onRowClick={setQuerySel} />
          <MatrixView title={`K (head ${h + 1})`} shape={`${n} × ${dK}`} M={hr.K} rowLabels={tokLabels} colLabels={dimLabels(dK, "k")} />
          <MatrixView title={`V (head ${h + 1})`} shape={`${n} × ${dK}`} M={hr.V} rowLabels={tokLabels} colLabels={dimLabels(dK, "v")} />
        </div>
      </div>
    );
  } else if (stage >= 3 && stage <= 6) {
    const M = stage === 3 ? hr.scores : stage === 4 ? hr.scaled : stage === 5 ? hr.masked : hr.weights;
    const title = stage === 3 ? "Raw scores S = Q Kᵀ" : stage === 4 ? `Scaled scores S / √${dK}` : stage === 5 ? (causal ? "Masked scores (j > i set to −∞)" : "Masked scores (no mask active)") : "Attention weights A = softmax(row-wise)";
    body = (
      <div className="grid gap-4">
        <MatrixView
          title={title}
          shape={`${n} × ${n}  (rows = queries, columns = keys) · head ${h + 1}`}
          M={M}
          rowLabels={tokLabels}
          colLabels={r.tokens}
          mode={stage === 6 ? "seq" : "signed"}
          digits={stage === 6 ? 3 : 2}
          highlightRow={q}
          onRowClick={setQuerySel}
          rowSums={stage === 6}
        />
        {stage === 6 && heads > 1 && (
          <div>
            <div className="mb-1 text-xs font-medium">Compare heads (click to select)</div>
            <div className="flex flex-wrap gap-3">
              {r.heads.map((hh, k) => (
                <button key={k} onClick={() => setHeadSel(k)} className="rounded-md border p-1.5" style={{ borderColor: k === h ? "var(--gold)" : "var(--border)" }} aria-label={`Show head ${k + 1}`}>
                  <MiniHeatmap M={hh.weights} />
                  <div className="mt-1 text-center text-[10px] text-muted">head {k + 1}</div>
                </button>
              ))}
            </div>
          </div>
        )}
        <div>
          <div className="mb-1 text-xs text-muted">Connections from the selected query (thickness ∝ final softmax weight, head {h + 1}). Click a query token above the lines.</div>
          <ConnectionDiagram tokens={r.tokens} weights={hr.weights} masked={hr.masked} q={q} onPick={setQuerySel} />
        </div>
      </div>
    );
  } else if (stage === 7) {
    body = (
      <div className="grid gap-4">
        <div className="grid gap-4 lg:grid-cols-2">
          <MatrixView title="A (weights)" shape={`${n} × ${n}`} M={hr.weights} rowLabels={tokLabels} colLabels={r.tokens} mode="seq" digits={2} highlightRow={q} onRowClick={setQuerySel} />
          <MatrixView title={`V (head ${h + 1})`} shape={`${n} × ${dK}`} M={hr.V} rowLabels={tokLabels} colLabels={dimLabels(dK, "v")} />
        </div>
        <MatrixView title={`Head output Z = A · V (head ${h + 1})`} shape={`${n} × ${dK}`} M={hr.out} rowLabels={tokLabels} colLabels={dimLabels(dK, "z")} highlightRow={q} onRowClick={setQuerySel} />
        <p className="font-mono text-[11px] text-muted">
          z[{q}] = {hr.weights[q].map((w, j) => `${w.toFixed(2)}·v${j}`).join(" + ")}
        </p>
        <ConnectionDiagram tokens={r.tokens} weights={hr.weights} masked={hr.masked} q={q} onPick={setQuerySel} />
      </div>
    );
  } else {
    body = (
      <div className="grid gap-4">
        <MatrixView
          title={heads > 1 ? `Concat(head 1, …, head ${heads})` : "Concat(head 1): a single head"}
          shape={`${n} × ${heads}·${dK} = ${n} × ${D_MODEL}`}
          M={r.concat}
          rowLabels={tokLabels}
          colLabels={Array.from({ length: D_MODEL }, (_, j) => `h${Math.floor(j / dK) + 1}.${j % dK}`)}
          highlightRow={q}
          onRowClick={setQuerySel}
          colGroups={heads > 1 ? dK : undefined}
        />
        <MatrixView title="Output Y = Concat · W_O" shape={`(${n} × ${D_MODEL}) · (${D_MODEL} × ${D_MODEL}) = ${n} × ${D_MODEL}`} M={r.Y} rowLabels={tokLabels} colLabels={dimLabels(D_MODEL, "y")} highlightRow={q} onRowClick={setQuerySel} />
        <div className="rounded-lg border border-border p-3 text-xs">
          <div className="mb-1 font-medium">Shapes and parameters (d_model = {D_MODEL}, h = {heads}, dₖ = d_v = {dK})</div>
          <ul className="list-disc space-y-0.5 pl-5 font-mono text-[11px] text-muted">
            <li>W_Q, W_K, W_V: {D_MODEL} × {D_MODEL} each (= {heads} heads × {D_MODEL} × {dK})</li>
            <li>W_O: {D_MODEL} × {D_MODEL}</li>
            <li>
              weights: 4·d² = {4 * D_MODEL * D_MODEL}; with biases 4d² + 4d = {mhaParamCount(D_MODEL)} (independent of h)
            </li>
            <li>score matrices: {heads} × ({n} × {n}) — memory grows as n²</li>
          </ul>
        </div>
      </div>
    );
  }

  return (
    <SimShell title="Attention & Transformer Explorer" subtitle="Scaled dot-product, multi-head self-attention on your own sentence, one exact step at a time." controls={controls} aside={aside}>
      <nav className="mb-4 flex flex-wrap gap-1" aria-label="Stages">
        {STAGES.map((name, k) => (
          <button
            key={k}
            onClick={() => s.setStep(k)}
            className="rounded-md border px-2 py-1 text-[11px]"
            style={{
              borderColor: k === stage ? "var(--accent)" : "var(--border)",
              background: k === stage ? "var(--accent-soft)" : "transparent",
              color: k === stage ? "var(--text)" : "var(--text-muted)",
            }}
            aria-current={k === stage ? "step" : undefined}
          >
            {k + 1}. {name}
          </button>
        ))}
      </nav>
      {body}
    </SimShell>
  );
}

function MiniHeatmap({ M }: { M: Matrix }) {
  const n = M.length;
  const c = 10;
  return (
    <svg viewBox={`0 0 ${n * c} ${n * c}`} width={Math.max(60, n * c)} height={Math.max(60, n * c)} role="img" aria-label="Attention weight heatmap">
      {M.map((row, i) => row.map((v, j) => <rect key={`${i}-${j}`} x={j * c} y={i * c} width={c - 1} height={c - 1} fill={seqColor(v)} />))}
    </svg>
  );
}

function stageNarration(stage: number, r: AttentionResult, h: number, q: number, o: { heads: number; usePE: boolean; causal: boolean }): ReactNode {
  const n = r.tokens.length;
  if (n === 0) return "Enter at least one word.";
  const hr = r.heads[h];
  const dK = r.dK;
  const qt = `“${r.tokens[q]}”`;
  const row = hr.weights[q];
  const best = row.indexOf(Math.max(...row));
  switch (stage) {
    case 0:
      return `The sentence is split on whitespace into n = ${n} tokens. Each token is looked up in a vocabulary and replaced by an integer ID. Attention never sees the strings, only vectors derived from these IDs.`;
    case 1:
      return o.usePE
        ? `Each ID indexes a row of the embedding table, giving E (${n} × ${D_MODEL}). Because attention ignores order, we add the sinusoidal positional encoding PE row by row: X = E + PE. Row ${q} of X is what ${qt} contributes.`
        : `Each ID indexes a row of the embedding table, giving E (${n} × ${D_MODEL}). With PE switched off, X = E: the model would produce the same output for any permutation of the words (up to the same permutation of rows).`;
    case 2:
      return `Three learned matrices project every token: Q = XW_Q asks "what am I looking for?", K = XW_K says "what do I contain?", V = XW_V carries "what I pass on". With h = ${o.heads}, each head works in dₖ = ${D_MODEL}/${o.heads} = ${dK} dimensions.`;
    case 3:
      return `Every query is compared with every key by a dot product: S[i][j] = q_i · k_j. The row for ${qt} is ${hr.scores[q].map((v) => v.toFixed(2)).join(", ")}. This costs n² · dₖ = ${n * n * dK} multiply-adds for this head.`;
    case 4:
      return `Divide by √dₖ = ${Math.sqrt(dK).toFixed(3)}. For random q, k with unit-variance entries, q·k has variance dₖ; scaling restores unit variance so softmax does not saturate as dₖ grows.`;
    case 5:
      return o.causal
        ? `Causal mask: position i may only attend to positions j ≤ i, so every cell above the diagonal is set to −∞. After softmax, e^(−∞) = 0, so future tokens receive exactly zero weight. ${qt} can see ${q + 1} key(s).`
        : `No mask is active, so this matrix equals the scaled scores. Turn on "Causal mask" to see the upper triangle become −∞ (a decoder / GPT-style model). Padding masks work the same way, on padded key columns.`;
    case 6:
      return `Softmax is applied to each row separately, so each row becomes a probability distribution over keys (the Σ column is 1). In head ${h + 1}, ${qt} puts its largest weight, ${row[best].toFixed(3)}, on “${r.tokens[best]}”.`;
    case 7:
      return `Each output row is a weighted average of the value rows: z_i = Σ_j α_ij v_j. Since the weights are non-negative and sum to 1, z_${q} lies inside the convex hull of the value vectors. Shape: (${n}×${n})·(${n}×${dK}) = ${n}×${dK}.`;
    default:
      return `The ${o.heads} head outputs (each ${n} × ${dK}) are concatenated back to ${n} × ${D_MODEL} and mixed by W_O. In a Transformer block this Y is added to X (residual connection), normalised, and passed to a feed-forward network.`;
  }
}
