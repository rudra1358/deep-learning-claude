// Live webcam: a WebSocket relay between the browser and the Python ML service.
//
//   browser  <-- ws /api/stream -->  this server  <-- ws /stream -->  ML service
//
// Every message is passed through unchanged (config, JPEG frames, results), EXCEPT the last one:
// when the browser sends {"type":"stop"}, the ML service answers {"type":"session","run":{...}}
// with the whole session (tracks, line crossings, counts, summary). We save that run in MongoDB
// and send the browser {"type":"saved","run_id":"...","summary":{...}} instead.
//
// Back-pressure still works end to end: the browser sends the next frame only after it gets the
// previous result, and we never queue frames ourselves.
import { WebSocket, WebSocketServer } from "ws";
import { config } from "./config.js";
import { Run } from "./models/Run.js";

const MAX_FRAME_BYTES = 4 * 2 ** 20;

export function attachStreamRelay(httpServer) {
  const wss = new WebSocketServer({ server: httpServer, path: "/api/stream", maxPayload: MAX_FRAME_BYTES });

  wss.on("connection", (browser) => {
    const upstream = new WebSocket(config.mlUrl.replace(/^http/, "ws") + "/stream", { maxPayload: 64 * 2 ** 20 });
    const pending = []; // messages that arrive before the ML connection is open

    const sendBrowser = (obj) => browser.readyState === WebSocket.OPEN && browser.send(JSON.stringify(obj));

    browser.on("message", (data, isBinary) => {
      if (upstream.readyState === WebSocket.OPEN) upstream.send(data, { binary: isBinary });
      else pending.push([data, isBinary]);
    });

    upstream.on("open", () => {
      for (const [data, isBinary] of pending) upstream.send(data, { binary: isBinary });
      pending.length = 0;
    });

    upstream.on("message", async (data, isBinary) => {
      if (!isBinary) {
        let msg = null;
        try {
          msg = JSON.parse(data.toString());
        } catch {
          /* not JSON: pass through */
        }
        if (msg?.type === "session") {
          if (!msg.run) return sendBrowser({ type: "saved", run_id: null, summary: null });
          try {
            const run = await Run.create(msg.run);
            return sendBrowser({ type: "saved", run_id: run.id, summary: msg.run.summary });
          } catch (err) {
            console.error("saving live session failed:", err);
            return sendBrowser({ type: "error", message: "could not save the session" });
          }
        }
      }
      if (browser.readyState === WebSocket.OPEN) browser.send(data, { binary: isBinary });
    });

    upstream.on("error", () => {
      sendBrowser({ type: "error", message: `ML service unavailable at ${config.mlUrl}` });
      browser.close();
    });
    upstream.on("close", () => browser.readyState === WebSocket.OPEN && browser.close());
    browser.on("close", () => upstream.readyState <= WebSocket.OPEN && upstream.terminate());
    browser.on("error", () => upstream.terminate());
  });

  return wss;
}
