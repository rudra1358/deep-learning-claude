import { useNavigate } from "react-router-dom";
import { useEffect, useRef, useState } from "react";
import { ApiError, detectVideo, errorMessage, getRun } from "../lib/api";
import { useObjectUrl, usePageTitle } from "../lib/hooks";
import { shortId } from "../lib/format";
import { FileDrop } from "../components/FileDrop";
import { LineEditor } from "../components/LineEditor";
import { ModelSelect, useModels } from "../components/ModelSelect";
import { Card, ErrorState, PageHeader, ProgressBar, Slider, Spinner, StatusBadge } from "../components/ui";

const VIDEO_ACCEPT = ".mp4,.avi,.mov,.mkv,.webm";
const POLL_MS = 1000;

/**
 * Video analysis.
 *   1. POST /api/detect/video (multipart: file, model, conf, line as JSON string) -> 202 + queued run
 *   2. poll GET /api/runs/{id} every second for status/progress
 *   3. on "done" -> redirect to /runs/{id}; on "failed" -> show run.error
 */
export default function VideoPage() {
  usePageTitle("Video analysis");
  const navigate = useNavigate();
  const models = useModels();
  const [file, setFile] = useState(null);
  const [model, setModel] = useState("");
  const [conf, setConf] = useState(0.25);
  const [line, setLine] = useState(null);
  const [drawing, setDrawing] = useState(false);
  const [aspect, setAspect] = useState(16 / 9);
  const [previewError, setPreviewError] = useState(false);

  const [uploading, setUploading] = useState(false);
  const [run, setRun] = useState(null);
  const [error, setError] = useState(null);
  const url = useObjectUrl(file);
  const videoRef = useRef(null);

  // poll the queued run until it finishes
  useEffect(() => {
    if (!run || run.status === "done" || run.status === "failed") return;
    const ctrl = new AbortController();
    const id = setTimeout(() => {
      getRun(run.id, { signal: ctrl.signal })
        .then((r) => {
          setError(null);
          setRun(r);
          if (r.status === "done") navigate(`/runs/${r.id}`);
        })
        .catch((e) => {
          if (ctrl.signal.aborted) return;
          if (e instanceof ApiError && e.status === 404) {
            setError(`Run #${shortId(run.id)} no longer exists.`);
            setRun(null);
            return;
          }
          setError(`Lost track of run #${shortId(run.id)}: ${errorMessage(e)} — retrying…`);
          setRun({ ...run }); // trigger another poll
        });
    }, POLL_MS);
    return () => {
      clearTimeout(id);
      ctrl.abort();
    };
  }, [run, navigate]);

  const submit = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const r = await detectVideo({ file, model, conf, line });
      setRun(r);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setUploading(false);
    }
  };

  const reset = () => {
    setRun(null);
    setFile(null);
    setLine(null);
    setError(null);
  };

  const processing = uploading || (run !== null && run.status !== "failed");

  return (
    <>
      <PageHeader
        title="Video analysis"
        description="Upload a video to detect, track and count objects in every frame. Optionally draw a counting line on the first frame."
      />
      <div className="grid gap-4 lg:grid-cols-[1fr_20rem]">
        <div className="min-w-0 space-y-4">
          {error && <ErrorState message={error} />}
          <Card title="First frame">
            {url ? (
              <>
                <div
                  className="relative w-full overflow-hidden rounded-md bg-black"
                  style={{ aspectRatio: String(aspect) }}
                >
                  <video
                    ref={videoRef}
                    src={`${url}#t=0.001`}
                    className="absolute inset-0 h-full w-full"
                    preload="auto"
                    muted
                    playsInline
                    aria-label="First frame of the selected video"
                    onLoadedMetadata={(e) => {
                      const v = e.currentTarget;
                      if (v.videoWidth && v.videoHeight) setAspect(v.videoWidth / v.videoHeight);
                      setPreviewError(false);
                    }}
                    onError={() => setPreviewError(true)}
                  />
                  <LineEditor
                    value={line}
                    onChange={setLine}
                    active={drawing && !processing}
                    onDone={() => setDrawing(false)}
                  />
                  {previewError && (
                    <div className="absolute inset-0 grid place-items-center p-4 text-center text-sm text-white/80">
                      Your browser cannot preview this format (the server can still process it). The line will be placed
                      on a 16:9 frame.
                    </div>
                  )}
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  <button
                    type="button"
                    className="btn"
                    disabled={processing}
                    aria-pressed={drawing}
                    onClick={() => setDrawing((d) => !d)}
                  >
                    {drawing ? "Cancel drawing" : line ? "Redraw counting line" : "Draw counting line"}
                  </button>
                  {line && (
                    <button type="button" className="btn" disabled={processing} onClick={() => setLine(null)}>
                      Clear line
                    </button>
                  )}
                  <span className="self-center text-xs text-muted">
                    {line ? `Line: ${line.map((v) => v.toFixed(2)).join(", ")}` : "No counting line (optional)"}
                  </span>
                </div>
              </>
            ) : (
              <FileDrop
                accept={VIDEO_ACCEPT}
                onFile={setFile}
                label="Choose a video"
                hint="MP4, AVI, MOV, MKV or WebM"
              />
            )}
          </Card>
        </div>

        <div className="space-y-4">
          <Card title="Settings">
            <div className="space-y-4">
              {file && (
                <div className="text-sm">
                  <div className="label">File</div>
                  <div className="truncate" title={file.name}>
                    {file.name}
                  </div>
                  <div className="text-xs text-muted">{(file.size / 2 ** 20).toFixed(1)} MB</div>
                  {!processing && (
                    <button type="button" className="mt-1 text-xs text-accent underline" onClick={reset}>
                      Choose another file
                    </button>
                  )}
                </div>
              )}
              <ModelSelect id="video-model" value={model} onChange={setModel} models={models} disabled={processing} />
              <Slider
                id="video-conf"
                label="Confidence"
                value={conf}
                min={0.05}
                max={0.95}
                step={0.05}
                onChange={setConf}
                disabled={processing}
              />
              <button
                type="button"
                className="btn btn-primary w-full"
                disabled={!file || !model || processing || drawing}
                onClick={submit}
              >
                {uploading ? <Spinner /> : null}
                {uploading ? "Uploading…" : "Analyse video"}
              </button>
            </div>
          </Card>

          {run && (
            <Card title={`Run #${shortId(run.id)}`} actions={<StatusBadge status={run.status} />}>
              {run.status === "failed" ? (
                <div className="space-y-3">
                  <ErrorState message={run.error ?? "Processing failed."} />
                  <button type="button" className="btn w-full" onClick={reset}>
                    Try another video
                  </button>
                </div>
              ) : (
                <div className="space-y-2">
                  <ProgressBar
                    value={run.progress}
                    label={run.status === "queued" ? "Waiting in queue" : "Processing frames"}
                  />
                  <p className="text-xs text-muted" role="status">
                    {run.status === "done"
                      ? "Done — opening results…"
                      : "You can leave this page; the run will appear in History."}
                  </p>
                </div>
              )}
            </Card>
          )}
        </div>
      </div>
    </>
  );
}
