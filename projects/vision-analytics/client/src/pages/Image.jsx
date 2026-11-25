import { Link } from "react-router-dom";
import { useEffect, useState } from "react";
import { detectImage, errorMessage } from "../lib/api";
import { useDebounced, useObjectUrl, usePageTitle } from "../lib/hooks";
import { shortId, sortedEntries } from "../lib/format";
import { colorForLabel } from "../lib/draw";
import { BarList } from "../components/charts";
import { DetectionImage } from "../components/DetectionImage";
import { FileDrop } from "../components/FileDrop";
import { ModelSelect, useModels } from "../components/ModelSelect";
import { Card, EmptyState, ErrorState, PageHeader, Slider, Spinner, TimingBreakdown } from "../components/ui";

const IMAGE_ACCEPT = "image/jpeg,image/png,image/webp,image/bmp";

/**
 * Image detection.
 * Every change of file / model / slider (debounced 350 ms) calls POST /api/detect/image with
 * save=false, so experimenting does not fill the history. "Save to history" repeats the same
 * request with save=true, which stores a run and returns its run_id.
 */
export default function ImagePage() {
  usePageTitle("Image detection");
  const models = useModels();
  const [file, setFile] = useState(null);
  const [model, setModel] = useState("");
  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.45);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);
  const [savedId, setSavedId] = useState(null);
  const url = useObjectUrl(file);

  // debounce primitives (not an object literal, which would be "new" every render)
  const dConf = useDebounced(conf, 350);
  const dIou = useDebounced(iou, 350);

  useEffect(() => {
    if (!file || !model) return;
    const ctrl = new AbortController();
    setLoading(true);
    setError(null);
    setSavedId(null);
    detectImage({ file, model, conf: dConf, iou: dIou, save: false }, { signal: ctrl.signal })
      .then((r) => setResult(r))
      .catch((e) => {
        if (!ctrl.signal.aborted) setError(errorMessage(e));
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setLoading(false);
      });
    return () => ctrl.abort();
  }, [file, model, dConf, dIou]);

  const save = async () => {
    if (!file) return;
    setSaving(true);
    setError(null);
    try {
      const r = await detectImage({ file, model, conf, iou, save: true });
      setResult(r);
      setSavedId(r.run_id);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      <PageHeader
        title="Image detection"
        description="Upload an image and tune the thresholds — results update live. Nothing is stored until you click Save."
      />
      <div className="grid gap-4 lg:grid-cols-[20rem_1fr]">
        <div className="space-y-4">
          <Card title="Input">
            <div className="space-y-4">
              <FileDrop
                accept={IMAGE_ACCEPT}
                onFile={(f) => {
                  setResult(null);
                  setFile(f);
                }}
                label={file ? file.name : "Choose an image"}
                hint="JPEG, PNG, WebP or BMP"
                compact={!!file}
              />
              <ModelSelect id="image-model" value={model} onChange={setModel} models={models} />
              <Slider
                id="image-conf"
                label="Confidence"
                value={conf}
                min={0.05}
                max={0.95}
                step={0.05}
                onChange={setConf}
              />
              <Slider id="image-iou" label="NMS IoU" value={iou} min={0.1} max={0.9} step={0.05} onChange={setIou} />
              <button
                type="button"
                className="btn btn-primary w-full"
                disabled={!result || saving || loading}
                onClick={save}
              >
                {saving ? <Spinner /> : null}
                Save to history
              </button>
              {savedId !== null && (
                <p role="status" className="text-sm text-success">
                  Saved as{" "}
                  <Link to={`/runs/${savedId}`} className="font-medium underline">
                    run #{shortId(savedId)}
                  </Link>
                  .
                </p>
              )}
            </div>
          </Card>
          {result && (
            <Card title="Timing">
              <TimingBreakdown timing={result.timing} />
              <p className="mt-2 text-xs text-muted">
                {result.model} on {result.provider} · {result.width}×{result.height}
              </p>
            </Card>
          )}
        </div>

        <div className="min-w-0 space-y-4">
          {error && <ErrorState message={error} />}
          <Card
            title="Result"
            actions={
              loading ? (
                <span role="status" className="flex items-center gap-2 text-xs text-muted">
                  <Spinner /> Detecting…
                </span>
              ) : result ? (
                <span className="text-xs text-muted">{result.detections.length} objects</span>
              ) : null
            }
          >
            {url && result ? (
              <DetectionImage
                src={url}
                alt={`Uploaded image ${file?.name ?? ""} with detections`}
                items={result.detections}
                imageWidth={result.width}
                imageHeight={result.height}
              />
            ) : url ? (
              <img src={url} alt="Uploaded preview" className="w-full rounded-md opacity-60" />
            ) : (
              <EmptyState>Upload an image to start.</EmptyState>
            )}
          </Card>

          {result && (
            <div className="grid gap-4 xl:grid-cols-[1fr_18rem]">
              <Card title={`Detections (${result.detections.length})`}>
                {result.detections.length === 0 ? (
                  <EmptyState>No objects above the confidence threshold.</EmptyState>
                ) : (
                  <div className="-mx-4 max-h-96 overflow-auto sm:mx-0">
                    <table className="table">
                      <thead className="sticky top-0 bg-surface">
                        <tr>
                          <th scope="col">#</th>
                          <th scope="col">Label</th>
                          <th scope="col" className="text-right">
                            Score
                          </th>
                          <th scope="col">Box (x1, y1, x2, y2)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.detections.map((d, i) => (
                          <tr key={i}>
                            <td className="text-muted tabular-nums">{i + 1}</td>
                            <td>
                              <span className="inline-flex items-center gap-1.5">
                                <span
                                  className="h-2 w-2 rounded-sm"
                                  style={{ background: colorForLabel(d.label) }}
                                  aria-hidden
                                />
                                {d.label}
                              </span>
                            </td>
                            <td className="text-right tabular-nums">{d.score.toFixed(3)}</td>
                            <td className="font-mono text-xs whitespace-nowrap text-muted">
                              {d.box.map((v) => Math.round(v)).join(", ")}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </Card>
              <Card title="Counts per class">
                <BarList data={sortedEntries(result.counts)} emptyText="Nothing detected." />
              </Card>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
