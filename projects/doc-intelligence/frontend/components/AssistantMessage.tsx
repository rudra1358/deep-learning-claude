import type { AskMode, AskOut, Highlight, MessageOut, Source, Suggestion, Timing } from "@/lib/api";
import { pageLabel, timingLine } from "@/lib/text";
import AnswerText from "./AnswerText";
import SourceCard, { Highlighted } from "./SourceCard";

// One turn of the conversation as the UI needs it. A fresh /api/ask response and a message
// reloaded from /api/chats/{id} have different shapes, so both are converted into this one.
export type Turn =
  | { role: "user"; text: string }
  | {
      role: "assistant";
      answer: string | null; // null = not found
      mode: AskMode;
      confidence: number | null;
      highlight: Highlight | null;
      suggestion: Suggestion | null;
      gated: boolean;
      sources: Source[];
      timing: Timing;
    };

export function turnFromAsk(r: AskOut): Turn {
  return {
    role: "assistant",
    answer: r.found ? r.answer : null,
    mode: r.mode,
    confidence: r.confidence ?? null,
    highlight: r.highlight ?? null,
    suggestion: r.suggestion ?? null,
    gated: r.gated ?? false,
    sources: r.sources,
    timing: r.timing,
  };
}

export function turnFromHistory(m: MessageOut): Turn {
  if (m.role === "user") return { role: "user", text: m.content };
  const meta = m.meta ?? {};
  return {
    role: "assistant",
    // the server stores a placeholder sentence as content when nothing was found, so trust meta.found
    answer: meta.found === false ? null : m.content,
    mode: meta.mode ?? "extractive",
    confidence: meta.confidence ?? null,
    highlight: meta.highlight ?? null,
    suggestion: meta.suggestion ?? null,
    gated: meta.gated ?? false,
    sources: m.sources ?? [],
    timing: meta.timing ?? {},
  };
}

type AssistantTurn = Extract<Turn, { role: "assistant" }>;

// The quoted sentence around an answer span, with the span highlighted.
function Quote({ source, highlight, label }: { source: Source; highlight: Highlight; label: string }) {
  return (
    <blockquote className="border-l-4 border-accent pl-3 text-sm text-muted">
      <p className="mb-1 text-xs">
        {label} source {source.n} · {source.filename} · {pageLabel(source)}
      </p>
      <p className="break-words text-fg">
        {highlight.start > 120 && "…"}
        <Highlighted
          text={source.text.slice(Math.max(0, highlight.start - 120), highlight.end + 120)}
          start={Math.min(120, highlight.start)}
          end={Math.min(120, highlight.start) + (highlight.end - highlight.start)}
        />
        {highlight.end + 120 < source.text.length && "…"}
      </p>
    </blockquote>
  );
}

// Renders the answer, its evidence (numbered source cards) and how long each stage took.
export default function AssistantMessage({ turn, idPrefix }: { turn: AssistantTurn; idPrefix: string }) {
  const { answer, mode, confidence, highlight, suggestion, gated, sources, timing } = turn;
  const cited = highlight ? sources[highlight.source - 1] : undefined; // the source holding the extractive answer
  const suggested = suggestion ? sources[suggestion.highlight.source - 1] : undefined;
  // which span to mark inside the source cards: the answer, or else the low-confidence suggestion
  const mark = highlight ?? suggestion?.highlight ?? null;

  return (
    <div className="space-y-3 rounded border border-border bg-card p-4">
      <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
        <span className="rounded border border-border px-1.5">{mode}</span>
        {/* The reader's score is a logit margin (span score - "no answer" score), not a probability. */}
        {confidence != null && answer && (
          <span title="reader score: best span minus 'no answer' (higher = more certain)">
            reader score +{confidence.toFixed(1)}
          </span>
        )}
      </div>

      {answer === null ? (
        <div>
          <p className="font-medium text-warn">Not found in your documents.</p>
          {gated && (
            <p className="text-xs text-muted">
              The extractive reader checked the sources first and found no answer, so the language model was
              not asked (small LLMs tend to invent answers to questions the documents cannot answer).
            </p>
          )}
        </div>
      ) : mode === "generative" ? (
        <AnswerText text={answer} idPrefix={idPrefix} sourceCount={sources.length} />
      ) : (
        <p className="text-lg font-semibold">{answer}</p>
      )}

      {/* Extractive answers are an exact quote: show them highlighted inside the sentence they came from. */}
      {answer !== null && cited && highlight && <Quote source={cited} highlight={highlight} label="From" />}

      {/* Not confident enough to answer - but show the best candidate, clearly labelled, so it is one click to check. */}
      {answer === null && suggestion && suggested && (
        <div className="space-y-2 rounded border border-dashed border-border p-3">
          <p className="text-sm">
            <span className="font-medium">Possible answer (low confidence): </span>
            <span className="font-semibold">{suggestion.text}</span>
          </p>
          <p className="text-xs text-muted">
            Reader score +{suggestion.confidence.toFixed(1)} is below the answer threshold +
            {suggestion.threshold.toFixed(1)} (tuned on held-out questions). Check the source before relying on it.
          </p>
          <Quote source={suggested} highlight={suggestion.highlight} label="In" />
        </div>
      )}

      {sources.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-muted">Sources</h3>
          {sources.map((s) => (
            <SourceCard
              key={s.n}
              id={`${idPrefix}-src-${s.n}`}
              source={s}
              span={mark && mark.source === s.n ? { start: mark.start, end: mark.end } : null}
            />
          ))}
        </div>
      )}

      {timingLine(timing) && <p className="text-xs text-muted">{timingLine(timing)}</p>}
    </div>
  );
}
