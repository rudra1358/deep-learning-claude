"use client";

import { useEffect, useRef, useState } from "react";
import { getRunFrames, runVideoUrl, type FramesFile } from "@/lib/api";
import { drawCountingLine, drawDetections, fitCanvas } from "@/lib/draw";
import { useAsync, useElementSize } from "@/lib/hooks";
import { ErrorState, LoadingState } from "./ui";

/**
 * Plays the stored upload (GET /api/runs/{id}/video) and draws the tracker output from
 * GET /api/runs/{id}/frames on top. A requestAnimationFrame loop reads video.currentTime,
 * picks frame index floor(currentTime * fps) and redraws only when the index (or size) changes.
 */
export function VideoOverlayPlayer({ runId, line }: { runId: number; line: [number, number, number, number] | null }) {
  const frames = useAsync<FramesFile>((signal) => getRunFrames(runId, { signal }), [runId]);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const [videoError, setVideoError] = useState<string | null>(null);
  const [showBoxes, setShowBoxes] = useState(true);
  const [aspect, setAspect] = useState<number | null>(null);

  const data = frames.data;
  const ratio = aspect ?? (data && data.height ? data.width / data.height : 16 / 9);

  useEffect(() => {
    if (!data) return;
    let raf = 0;
    let lastIdx = -1;
    let lastW = 0;
    let lastShow = !showBoxes;
    const tick = () => {
      const v = videoRef.current;
      const c = canvasRef.current;
      if (v && c && size.width > 0) {
        const n = data.frames.length;
        const idx = n ? Math.min(n - 1, Math.max(0, Math.floor(v.currentTime * data.fps))) : -1;
        if (idx !== lastIdx || size.width !== lastW || showBoxes !== lastShow) {
          lastIdx = idx;
          lastW = size.width;
          lastShow = showBoxes;
          const ctx = fitCanvas(c, size.width, size.height);
          if (ctx) {
            const f = idx >= 0 ? data.frames[idx] : undefined;
            if (f && showBoxes) drawDetections(ctx, f.tracks, size.width / data.width, size.height / data.height);
            if (line) drawCountingLine(ctx, line, size.width, size.height);
          }
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [data, size, line, showBoxes]);

  if (frames.loading && !data) return <LoadingState label="Loading tracking data…" />;
  if (frames.error) return <ErrorState message={`Tracking data unavailable: ${frames.error}`} onRetry={frames.reload} />;

  return (
    <div>
      <div ref={wrapRef} className="relative w-full overflow-hidden rounded-md bg-black" style={{ aspectRatio: String(ratio) }}>
        <video
          ref={videoRef}
          src={runVideoUrl(runId)}
          controls
          playsInline
          preload="metadata"
          className="absolute inset-0 h-full w-full"
          aria-label="Analysed video with tracking overlay"
          onLoadedMetadata={(e) => {
            const v = e.currentTarget;
            if (v.videoWidth && v.videoHeight) setAspect(v.videoWidth / v.videoHeight);
          }}
          onError={() =>
            setVideoError(
              "The browser could not play this video (codec/container not supported, e.g. AVI/MKV). The analytics below are still valid.",
            )
          }
        />
        {/* pointer-events-none: clicks pass through to the native video controls */}
        <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden />
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-muted">
        <label className="inline-flex items-center gap-1.5">
          <input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} />
          Show boxes
        </label>
        {data && (
          <span>
            {data.frames.length} frames · {data.fps.toFixed(1)} fps · {data.width}×{data.height}
          </span>
        )}
      </div>
      {videoError && (
        <div className="mt-2">
          <ErrorState message={videoError} />
        </div>
      )}
    </div>
  );
}
