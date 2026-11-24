"use client";

import { useState } from "react";
import { CheckCircle2, XCircle, RotateCcw } from "lucide-react";
import type { QuizQuestion } from "@/lib/types";
import { Inline } from "./Inline";
import { progress } from "@/lib/progress";

export function Quiz({ questions, quizKey }: { questions: QuizQuestion[]; quizKey: string }) {
  const [answers, setAnswers] = useState<(number | string | null)[]>(() => questions.map(() => null));
  const [submitted, setSubmitted] = useState(false);

  const correct = (q: QuizQuestion, a: number | string | null) => {
    if (a === null || a === "") return false;
    if (q.kind === "numeric") {
      const v = parseFloat(String(a));
      return Number.isFinite(v) && Math.abs(v - q.answer) <= q.tolerance;
    }
    return a === q.answer;
  };

  const score = questions.reduce((s, q, i) => s + (correct(q, answers[i]) ? 1 : 0), 0);

  function submit() {
    setSubmitted(true);
    progress.recordQuiz(quizKey, score, questions.length);
  }

  return (
    <div className="my-6 space-y-5">
      {questions.map((q, qi) => {
        const a = answers[qi];
        const ok = correct(q, a);
        return (
          <div key={qi} className="card p-5">
            <div className="mb-3 flex gap-2 font-medium">
              <span className="text-faint">{qi + 1}.</span>
              <div className="min-w-0 flex-1">
                <Inline text={q.q} />
              </div>
            </div>
            {q.kind === "numeric" ? (
              <input
                type="text"
                inputMode="decimal"
                className="field w-40"
                placeholder="Your answer"
                value={(a as string) ?? ""}
                disabled={submitted}
                onChange={(e) => setAnswers((prev) => prev.map((x, i) => (i === qi ? e.target.value : x)))}
                aria-label={`Answer to question ${qi + 1}`}
              />
            ) : (
              <div className="grid gap-2">
                {q.options.map((opt, oi) => {
                  const chosen = a === oi;
                  let cls = "border-border hover:bg-surface-2";
                  if (submitted && oi === q.answer) cls = "border-accent bg-accent-soft";
                  else if (submitted && chosen) cls = "border-clay bg-clay-soft";
                  else if (chosen) cls = "border-accent bg-accent-soft";
                  return (
                    <button
                      key={oi}
                      disabled={submitted}
                      onClick={() => setAnswers((prev) => prev.map((x, i) => (i === qi ? oi : x)))}
                      className={`rounded-lg border px-3 py-2 text-left text-sm transition-colors ${cls}`}
                    >
                      <Inline text={opt} />
                    </button>
                  );
                })}
              </div>
            )}
            {submitted && (
              <div className={`mt-3 flex gap-2 rounded-lg p-3 text-sm ${ok ? "bg-accent-soft" : "bg-clay-soft"}`}>
                {ok ? <CheckCircle2 size={18} className="shrink-0 text-accent" /> : <XCircle size={18} className="shrink-0 text-clay" />}
                <div className="min-w-0 flex-1">
                  {q.kind === "numeric" && !ok && <div className="mb-1">Correct answer: {q.answer}</div>}
                  <Inline text={q.explain} />
                </div>
              </div>
            )}
          </div>
        );
      })}
      <div className="flex items-center gap-3">
        {!submitted ? (
          <button className="sim-btn primary" onClick={submit} disabled={answers.some((a) => a === null || a === "")}>
            Check answers
          </button>
        ) : (
          <>
            <span className="text-sm">
              Score: <strong>{score}</strong> / {questions.length}
            </span>
            <button
              className="sim-btn"
              onClick={() => {
                setSubmitted(false);
                setAnswers(questions.map(() => null));
              }}
            >
              <RotateCcw size={14} /> Retry
            </button>
          </>
        )}
      </div>
    </div>
  );
}
