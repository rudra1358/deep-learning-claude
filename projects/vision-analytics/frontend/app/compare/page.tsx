"use client";

import { useState } from "react";
import { compareModels, errorMessage, type CompareResult, type ImageResult } from "@/lib/api";
import { useObjectUrl } from "@/lib/hooks";
import { sortedEntries, sumValues } from "@/lib/format";
import { BarList } from "@/components/charts";
import { DetectionImage } from "@/components/DetectionImage";
import { FileDrop } from "@/components/FileDrop";
import { ModelSelect, useModels } from "@/components/ModelSelect";
import { Card, EmptyState, ErrorState, PageHeader, Slider, Spinner, TimingBreakdown } from "@/components/ui";

/**
 * POST /api/compare with models="a,b": the server runs the same image through both models
 * (with one untimed warm-up run each, so latency is fair) and returns one ImageResult per model.
 */
export default function ComparePage() {
  const models = useModels();
  const [file, setFile] = useState<File | null>(null);
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.45);
  const [res, setRes] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const url = useObjectUrl(file);

  const tooFewModels = !models.loading && (models.data?.length ?? 0) < 2;

  const run = async () => {
    if (!file || !a || !b) return;
    setLoading(true);
    setError(null);
    try {
      setRes(await compareModels({ file, models: [a, b], conf, iou }));
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setLoading(false);
    }
  };

  const [ra, rb] = res?.results ?? [];

  return (
    <>
      <PageHeader title="Compare models" description="Run one image through two models and compare what they find and how fast they are." />

      <Card className="mb-4">
        <div className="grid gap-4 md:grid-cols-[1fr_1fr_1fr] md:items-end">
          <FileDrop
            accept="image/jpeg,image/png,image/webp,image/bmp"
            onFile={(f) => {
              setFile(f);
              setRes(null);
            }}
            label={file ? file.name : "Choose an image"}
            compact
          />
          <div className="space-y-3">
            <ModelSelect id="cmp-a" label="Model A" value={a} onChange={setA} models={models} />
            <ModelSelect id="cmp-b" label="Model B" value={b} onChange={setB} models={models} exclude={a} />
          </div>
          <div className="space-y-3">
            <Slider id="cmp-conf" label="Confidence" value={conf} min={0.05} max={0.95} step={0.05} onChange={setConf} />
            <Slider id="cmp-iou" label="NMS IoU" value={iou} min={0.1} max={0.9} step={0.05} onChange={setIou} />
            <button type="button" className="btn btn-primary w-full" disabled={!file || !a || !b || a === b || loading} onClick={run}>
              {loading && <Spinner />}
              {loading ? "Comparing…" : "Compare"}
            </button>
          </div>
        </div>
        {tooFewModels && <p className="mt-3 text-sm text-warning">At least two models are needed on the server to compare.</p>}
      </Card>

      {error && (
        <div className="mb-4">
          <ErrorState message={error} />
        </div>
      )}

      {!res && !loading && <EmptyState>Pick an image and two models, then click Compare.</EmptyState>}

      {ra && rb && url && (
        <>
          <DeltaRow a={ra} b={rb} />
          <div className="mt-4 grid gap-4 lg:grid-cols-2">
            {[ra, rb].map((r, i) => (
              <Card key={`${r.model}-${i}`} title={`${i === 0 ? "A" : "B"} · ${r.model}`} actions={<span className="text-xs text-muted">{r.provider}</span>}>
                <DetectionImage
                  src={url}
                  alt={`Detections by ${r.model}`}
                  items={r.detections}
                  imageWidth={r.width}
                  imageHeight={r.height}
                />
                <div className="mt-4 space-y-4">
                  <TimingBreakdown timing={r.timing} />
                  <div>
                    <div className="label">Counts ({r.detections.length})</div>
                    <BarList data={sortedEntries(r.counts)} emptyText="Nothing detected." />
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </>
      )}
    </>
  );
}

/** B minus A for total objects, latency and every class. */
function DeltaRow({ a, b }: { a: ImageResult; b: ImageResult }) {
  const labels = Array.from(new Set([...Object.keys(a.counts), ...Object.keys(b.counts)])).sort();
  const dObj = sumValues(b.counts) - sumValues(a.counts);
  const dMs = b.timing.total_ms - a.timing.total_ms;
  const sign = (n: number, digits = 0) => `${n > 0 ? "+" : ""}${n.toFixed(digits)}`;
  const speed = a.timing.total_ms > 0 ? b.timing.total_ms / a.timing.total_ms : null;

  return (
    <Card title="Delta (B − A)">
      <div className="-mx-4 overflow-x-auto sm:mx-0">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Metric</th>
              <th scope="col" className="text-right">
                A
              </th>
              <th scope="col" className="text-right">
                B
              </th>
              <th scope="col" className="text-right">
                Δ
              </th>
            </tr>
          </thead>
          <tbody>
            <tr className="font-medium">
              <td>Total latency</td>
              <td className="text-right tabular-nums">{a.timing.total_ms.toFixed(1)} ms</td>
              <td className="text-right tabular-nums">{b.timing.total_ms.toFixed(1)} ms</td>
              <td className={`text-right tabular-nums ${dMs > 0 ? "text-danger" : "text-success"}`}>
                {sign(dMs, 1)} ms{speed ? ` (${speed.toFixed(2)}×)` : ""}
              </td>
            </tr>
            <tr className="font-medium">
              <td>Objects</td>
              <td className="text-right tabular-nums">{sumValues(a.counts)}</td>
              <td className="text-right tabular-nums">{sumValues(b.counts)}</td>
              <td className="text-right tabular-nums">{sign(dObj)}</td>
            </tr>
            {labels.map((l) => {
              const va = a.counts[l] ?? 0;
              const vb = b.counts[l] ?? 0;
              return (
                <tr key={l}>
                  <td className="text-muted">{l}</td>
                  <td className="text-right tabular-nums">{va}</td>
                  <td className="text-right tabular-nums">{vb}</td>
                  <td className="text-right tabular-nums">{sign(vb - va)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
