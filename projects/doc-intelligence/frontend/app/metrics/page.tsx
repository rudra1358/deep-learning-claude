"use client";

import { useCallback, useEffect, useState } from "react";
import { JsonBlock } from "@/components/MetricsView";
import { ErrorBox, Loading } from "@/components/ui";
import { errorMessage, getMetrics, type Json, type Metrics } from "@/lib/api";

// The sections the training scripts write, with what each one measures and which script produces it.
const SECTIONS: { key: string; title: string; about: string; script: string }[] = [
  {
    key: "retrieval",
    title: "Retrieval",
    about: "recall@k = share of questions where a chunk containing the answer is in the top k; MRR@10 = average of 1/rank of the first such chunk. Compares BM25, dense, hybrid and hybrid+rerank, plus a chunk-size sweep.",
    script: "training/eval_retrieval.py",
  },
  {
    key: "qa",
    title: "Question answering (SQuAD 2.0)",
    about: "EM = exact match, F1 = word overlap with the gold answer. 'oracle' is given the correct paragraph; 'system' must first retrieve it - the gap is the cost of retrieval.",
    script: "training/eval_qa.py",
  },
  {
    key: "summarization",
    title: "Summarization (CNN/DailyMail)",
    about: "ROUGE = n-gram overlap with human-written summaries. lead-3 (the first three sentences) is the classic baseline to beat.",
    script: "training/eval_summarization.py",
  },
  {
    key: "ner",
    title: "Named-entity recognition (CoNLL-2003)",
    about: "Entity-level precision / recall / F1 per type for our fine-tuned DistilBERT.",
    script: "training/finetune_ner.py",
  },
  {
    key: "classifier",
    title: "Topic classifier (AG News)",
    about: "Accuracy and macro-F1 of our fine-tuned DistilBERT, with the confusion matrix.",
    script: "training/finetune_classifier.py",
  },
  {
    key: "generation",
    title: "Generative answers (RAG)",
    about: "How often the LLM's answer contains the gold answer and cites a source, and how often it correctly refuses unanswerable questions.",
    script: "training/eval_generation.py",
  },
];

// The QA numbers are nested deep in the JSON; pull out the headline "oracle vs system" comparison.
function qaHeadline(qa: Json | undefined): Json | null {
  if (!qa || typeof qa !== "object" || Array.isArray(qa)) return null;
  const run = qa.squad2;
  if (!run || typeof run !== "object" || Array.isArray(run)) return null;
  const pick = (name: string): Json => {
    const r = run[name];
    return r && typeof r === "object" && !Array.isArray(r) ? (r.tuned_on_first_half_reported_on_second ?? null) : null;
  };
  return { "oracle (gold paragraph)": pick("oracle"), "system (retrieve + read)": pick("system") };
}

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState("");

  const load = useCallback(() => {
    setError("");
    getMetrics()
      .then(setMetrics)
      .catch((e) => setError(errorMessage(e)));
  }, []);

  useEffect(load, [load]);

  if (error) return <ErrorBox message={error} onRetry={load} />;
  if (!metrics) return <Loading text="Loading metrics…" />;

  // Sections the UI does not know about yet are still shown, after the known ones.
  const extra = Object.keys(metrics).filter((k) => !SECTIONS.some((s) => s.key === k));
  const all = [...SECTIONS, ...extra.map((k) => ({ key: k, title: k, about: "", script: "" }))];
  const headline = qaHeadline(metrics.qa);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Metrics</h1>
        <p className="text-sm text-muted">Offline evaluation results written by the training scripts (training/reports/metrics.json).</p>
      </div>
      {all.map((s) => (
        <section key={s.key} aria-labelledby={`sec-${s.key}`} className="space-y-3 rounded border border-border bg-card p-4">
          <h2 id={`sec-${s.key}`} className="text-lg font-semibold">
            {s.title}
          </h2>
          {s.about && <p className="text-sm text-muted">{s.about}</p>}
          {metrics[s.key] === undefined ? (
            <p className="text-sm text-warn">
              Not run yet{s.script && <> - run <code>{s.script}</code></>}.
            </p>
          ) : (
            <>
              {s.key === "qa" && headline && (
                <div>
                  <h3 className="mb-1 text-sm font-semibold">Headline: tuned threshold, held-out half</h3>
                  <JsonBlock value={headline} />
                </div>
              )}
              <JsonBlock value={metrics[s.key] ?? null} />
            </>
          )}
        </section>
      ))}
    </div>
  );
}
