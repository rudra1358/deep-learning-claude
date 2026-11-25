import { Link } from "react-router-dom";
import { useCallback, useEffect, useRef, useState } from "react";
import { streamUrl } from "../lib/api";
import { drawCountingLine, drawDetections, fitCanvas } from "../lib/draw";
import { useElementSize, usePageTitle } from "../lib/hooks";
import { shortId, sortedEntries, sumValues } from "../lib/format";
import { BarList } from "../components/charts";
import { LineEditor } from "../components/LineEditor";
import { ModelSelect, useModels } from "../components/ModelSelect";
import { Card, EmptyState, ErrorState, PageHeader, Slider, Spinner, StatCard, TimingBreakdown } from "../components/ui";

/** Max width of the JPEG frames sent to the server (smaller = faster upload + decode). */
const MAX_SEND_WIDTH = 640;
const JPEG_QUALITY = 0.7;

// phase: "idle" | "camera" | "connecting" | "running" | "stopping"

function cameraErrorMessage(e) {
  if (e instanceof DOMException) {
    switch (e.name) {
      case "NotAllowedError":
      case "SecurityError":
        return "Camera permission was denied. Allow camera access for this site in your browser settings and try again.";
      case "NotFoundError":
      case "OverconstrainedError":
        return "No camera was found on this device.";
      case "NotReadableError":
        return "The camera is already in use by another application.";
    }
  }
  return `Could not start the camera: ${e instanceof Error ? e.message : String(e)}`;
}

/**
 * Live webcam analysis over WebSocket (protocol in ml-service/app/routers/stream.py; the Express server relays it):
 *   1. send {"type":"config",...}  -> wait for {"type":"ready"}
 *   2. send one JPEG frame (binary) -> wait for {"type":"result"} -> send the next frame
 *      (only ever ONE frame in flight = back-pressure; a slow server just lowers the FPS)
 *   3. send {"type":"stop"}         -> server saves the session, replies {"type":"saved","run_id"}
 */
export default function LivePage() {
  usePageTitle("Live camera");
  const models = useModels();
  const [model, setModel] = useState("");
  const [conf, setConf] = useState(0.3);
  const [line, setLine] = useState(null);
  const [drawing, setDrawing] = useState(false);

  const [phase, setPhase] = useState("idle");
  const [error, setError] = useState(null);
  const [last, setLast] = useState(null);
  const [provider, setProvider] = useState(null);
  const [crossIn, setCrossIn] = useState({});
  const [crossOut, setCrossOut] = useState({});
  const [saved, setSaved] = useState(null);
  const [clientFps, setClientFps] = useState(0);

  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const grabRef = useRef(null);
  const streamRef = useRef(null);
  const wsRef = useRef(null);
  const phaseRef = useRef("idle");
  const sendSize = useRef({ w: 640, h: 480 });
  const resultTimes = useRef([]);
  const lastResultRef = useRef(null);
  const stopTimer = useRef(null);
  const [videoWrapRef, videoSize] = useElementSize();
  const [aspect, setAspect] = useState(4 / 3);

  const setPhaseBoth = (p) => {
    phaseRef.current = p;
    setPhase(p);
  };

  // ------------------------------------------------------------------ overlay drawing
  const drawOverlay = useCallback(() => {
    const c = overlayRef.current;
    if (!c || videoSize.width === 0) return;
    const ctx = fitCanvas(c, videoSize.width, videoSize.height);
    if (!ctx) return;
    const r = lastResultRef.current;
    if (r && (phaseRef.current === "running" || phaseRef.current === "stopping")) {
      // boxes are in the pixel space of the frames we SENT (sendSize), not the camera's native size
      drawDetections(ctx, r.tracks, videoSize.width / sendSize.current.w, videoSize.height / sendSize.current.h);
    }
    if (line && !drawing) drawCountingLine(ctx, line, videoSize.width, videoSize.height);
  }, [videoSize, line, drawing]);

  useEffect(drawOverlay, [drawOverlay, last, phase]);

  // ------------------------------------------------------------------ camera
  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const startCamera = async () => {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError(
        "Camera access needs a secure context (https:// or http://localhost) and a browser that supports getUserMedia.",
      );
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      const v = videoRef.current;
      if (!v) return;
      v.srcObject = stream;
      await v.play();
      if (v.videoWidth && v.videoHeight) setAspect(v.videoWidth / v.videoHeight);
      setPhaseBoth("camera");
    } catch (e) {
      stopCamera();
      setError(cameraErrorMessage(e));
    }
  };

  // ------------------------------------------------------------------ frame sending
  const sendFrame = useCallback(() => {
    const ws = wsRef.current;
    const v = videoRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !v || phaseRef.current !== "running") return;
    const { w, h } = sendSize.current;
    const canvas = (grabRef.current ??= document.createElement("canvas"));
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(v, 0, 0, w, h);
    canvas.toBlob(
      (blob) => {
        if (!blob || phaseRef.current !== "running" || ws.readyState !== WebSocket.OPEN) return;
        blob.arrayBuffer().then((buf) => {
          if (phaseRef.current === "running" && ws.readyState === WebSocket.OPEN) ws.send(buf);
        });
      },
      "image/jpeg",
      JPEG_QUALITY,
    );
  }, []);

  const closeSocket = useCallback(() => {
    if (stopTimer.current) clearTimeout(stopTimer.current);
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws && ws.readyState <= WebSocket.OPEN) ws.close();
  }, []);

  // ------------------------------------------------------------------ session
  const startSession = () => {
    const v = videoRef.current;
    if (!v || !v.videoWidth) {
      setError("Camera is not ready yet.");
      return;
    }
    setError(null);
    setSaved(null);
    setLast(null);
    lastResultRef.current = null;
    setCrossIn({});
    setCrossOut({});
    resultTimes.current = [];
    setDrawing(false);

    const w = Math.min(MAX_SEND_WIDTH, v.videoWidth);
    const h = Math.round((v.videoHeight * w) / v.videoWidth);
    sendSize.current = { w, h };

    setPhaseBoth("connecting");
    let ws;
    try {
      ws = new WebSocket(streamUrl());
    } catch (e) {
      setPhaseBoth("camera");
      setError(`Could not open WebSocket: ${e instanceof Error ? e.message : String(e)}`);
      return;
    }
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;
    let consecutiveErrors = 0;

    ws.onopen = () => {
      const cfg = { type: "config", model: model || null, conf, line, width: w, height: h };
      ws.send(JSON.stringify(cfg));
    };

    ws.onmessage = (ev) => {
      if (typeof ev.data !== "string") return;
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      switch (msg.type) {
        case "ready":
          setProvider(msg.provider);
          setPhaseBoth("running");
          sendFrame();
          break;
        case "result": {
          consecutiveErrors = 0;
          lastResultRef.current = msg;
          setLast(msg);
          const now = performance.now();
          resultTimes.current = [...resultTimes.current.filter((t) => now - t < 2000), now];
          setClientFps(resultTimes.current.length / 2);
          if (msg.events.length) {
            const add = (prev, dir) => {
              const next = { ...prev };
              for (const e of msg.events) if (e.direction === dir) next[e.label] = (next[e.label] ?? 0) + 1;
              return next;
            };
            setCrossIn((p) => add(p, "in"));
            setCrossOut((p) => add(p, "out"));
          }
          sendFrame(); // back-pressure: next frame only after this result arrived
          break;
        }
        case "saved":
          if (msg.run_id !== null) setSaved({ runId: msg.run_id, summary: msg.summary });
          closeSocket();
          setPhaseBoth("camera");
          break;
        case "error":
          setError(`Server: ${msg.message}`);
          if (phaseRef.current === "connecting") {
            // e.g. model not found -> config rejected
            closeSocket();
            setPhaseBoth("camera");
          } else if (phaseRef.current === "running") {
            consecutiveErrors += 1;
            if (consecutiveErrors < 5) sendFrame();
            else {
              closeSocket();
              setPhaseBoth("camera");
            }
          }
          break;
      }
    };

    ws.onerror = () => {
      setError(`WebSocket error — is the API reachable at ${streamUrl()}?`);
    };

    ws.onclose = (ev) => {
      if (wsRef.current !== ws) return; // we closed it on purpose
      wsRef.current = null;
      if (phaseRef.current === "running" || phaseRef.current === "connecting" || phaseRef.current === "stopping") {
        setError((prev) => prev ?? `Connection closed unexpectedly (code ${ev.code}).`);
        setPhaseBoth(streamRef.current ? "camera" : "idle");
      }
    };
  };

  const stopSession = () => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      closeSocket();
      setPhaseBoth("camera");
      return;
    }
    const processed = lastResultRef.current !== null;
    setPhaseBoth("stopping");
    ws.send(JSON.stringify({ type: "stop" }));
    if (!processed) {
      // nothing processed -> nothing to save; don't wait for the server's empty "saved" reply
      closeSocket();
      setPhaseBoth("camera");
      return;
    }
    stopTimer.current = setTimeout(() => {
      setError("Server did not confirm the save within 15 s.");
      closeSocket();
      setPhaseBoth("camera");
    }, 15000);
  };

  const turnOffCamera = () => {
    closeSocket();
    stopCamera();
    setPhaseBoth("idle");
  };

  // cleanup on unmount
  useEffect(
    () => () => {
      phaseRef.current = "idle";
      closeSocket();
      stopCamera();
    },
    [closeSocket, stopCamera],
  );

  const busy = phase === "connecting" || phase === "running" || phase === "stopping";
  const occupancy = last?.occupancy ?? {};

  return (
    <>
      <PageHeader
        title="Live camera"
        description="Frames from your webcam are sent to the API over a WebSocket; boxes, track IDs and trails are drawn on top in real time."
      />
      <div className="grid gap-4 lg:grid-cols-[1fr_20rem]">
        <div className="min-w-0 space-y-4">
          {error && <ErrorState message={error} />}
          <Card
            title="Camera"
            actions={
              phase === "running" ? (
                <span className="flex items-center gap-1.5 text-xs font-medium text-danger">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-danger" aria-hidden /> LIVE
                </span>
              ) : null
            }
          >
            <div
              ref={videoWrapRef}
              className="relative w-full overflow-hidden rounded-md bg-black"
              style={{ aspectRatio: String(aspect) }}
            >
              <video
                ref={videoRef}
                className="absolute inset-0 h-full w-full"
                playsInline
                muted
                aria-label="Webcam preview"
              />
              <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden />
              {phase !== "idle" && (drawing || (!busy && line)) && (
                <LineEditor value={line} onChange={setLine} active={drawing} onDone={() => setDrawing(false)} />
              )}
              {phase === "idle" && (
                <div className="absolute inset-0 grid place-items-center p-4 text-center text-sm text-white/80">
                  Camera is off. Click “Start camera” to begin.
                </div>
              )}
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {phase === "idle" ? (
                <button type="button" className="btn btn-primary" onClick={startCamera}>
                  Start camera
                </button>
              ) : (
                <>
                  {!busy ? (
                    <button
                      type="button"
                      className="btn btn-primary"
                      onClick={startSession}
                      disabled={drawing || !model}
                    >
                      Start analysis
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="btn btn-danger"
                      onClick={stopSession}
                      disabled={phase === "stopping"}
                    >
                      {phase === "stopping" || phase === "connecting" ? <Spinner /> : null}
                      {phase === "connecting" ? "Connecting…" : phase === "stopping" ? "Saving…" : "Stop & save"}
                    </button>
                  )}
                  <button
                    type="button"
                    className="btn"
                    onClick={() => setDrawing((d) => !d)}
                    disabled={busy}
                    aria-pressed={drawing}
                  >
                    {drawing ? "Cancel drawing" : line ? "Redraw counting line" : "Draw counting line"}
                  </button>
                  {line && !busy && (
                    <button type="button" className="btn" onClick={() => setLine(null)}>
                      Clear line
                    </button>
                  )}
                  <button type="button" className="btn" onClick={turnOffCamera}>
                    Turn off camera
                  </button>
                </>
              )}
            </div>
            {busy && (
              <p className="mt-2 text-xs text-muted">
                Stop the session to change model, confidence or the counting line.
              </p>
            )}
          </Card>

          {saved && (
            <div role="status" className="rounded-md border border-success/40 bg-success/10 p-3 text-sm">
              Session saved as{" "}
              <Link to={`/runs/${saved.runId}`} className="font-semibold text-success underline">
                run #{shortId(saved.runId)}
              </Link>{" "}
              — {saved.summary.frames ?? 0} frames, {sumValues(saved.summary.unique_objects)} unique objects.
            </div>
          )}
        </div>

        <div className="space-y-4">
          <Card title="Settings">
            <div className="space-y-4">
              <ModelSelect id="live-model" value={model} onChange={setModel} models={models} disabled={busy} />
              <Slider
                id="live-conf"
                label="Confidence"
                value={conf}
                min={0.05}
                max={0.95}
                step={0.05}
                onChange={setConf}
                disabled={busy}
              />
              <p className="text-xs text-muted">
                Counting line: {line ? line.map((v) => v.toFixed(2)).join(", ") : "none"}
              </p>
            </div>
          </Card>

          <div className="grid grid-cols-2 gap-3">
            <StatCard label="Server FPS" value={last ? last.server_fps.toFixed(1) : "—"} hint={provider ?? undefined} />
            <StatCard
              label="Client FPS"
              value={last ? clientFps.toFixed(1) : "—"}
              hint={last ? `frame ${last.frame}` : undefined}
            />
          </div>

          <Card title="Latency (last frame)">
            {last ? <TimingBreakdown timing={last.timing} /> : <EmptyState>Start analysis to see timings.</EmptyState>}
          </Card>

          <Card title="In view now">
            <BarList data={sortedEntries(occupancy)} emptyText={last ? "Nothing in view." : "No data yet."} />
          </Card>

          <Card title="Line crossings (cumulative)">
            {!line && !last ? (
              <EmptyState>Draw a counting line before starting to count crossings.</EmptyState>
            ) : (
              <CrossingTable crossIn={crossIn} crossOut={crossOut} />
            )}
          </Card>
        </div>
      </div>
    </>
  );
}

function CrossingTable({ crossIn, crossOut }) {
  const labels = Array.from(new Set([...Object.keys(crossIn), ...Object.keys(crossOut)])).sort();
  if (labels.length === 0) return <p className="text-sm text-muted">No crossings yet.</p>;
  return (
    <table className="table">
      <thead>
        <tr>
          <th scope="col">Label</th>
          <th scope="col" className="text-right">
            In
          </th>
          <th scope="col" className="text-right">
            Out
          </th>
        </tr>
      </thead>
      <tbody>
        {labels.map((l) => (
          <tr key={l}>
            <td>{l}</td>
            <td className="text-right tabular-nums">{crossIn[l] ?? 0}</td>
            <td className="text-right tabular-nums">{crossOut[l] ?? 0}</td>
          </tr>
        ))}
        <tr className="font-medium">
          <td>Total</td>
          <td className="text-right tabular-nums">{sumValues(crossIn)}</td>
          <td className="text-right tabular-nums">{sumValues(crossOut)}</td>
        </tr>
      </tbody>
    </table>
  );
}
