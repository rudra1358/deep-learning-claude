// Video jobs, seen from the server.
//
//   1. POST /api/detect/video saves the upload and creates a run {status: "queued"} in MongoDB.
//   2. startVideoJob() asks the ML service to process it (POST /jobs/video).
//   3. watchJob() polls the ML service (GET /jobs/<run id>) every second and copies status and
//      progress into the run, so the browser - which polls GET /api/runs/<id> - sees it move.
//   4. When the job is done, the result (tracks, events, counts, summary) is saved into the run
//      and the ML service is told to forget the job (DELETE /jobs/<run id>).
//
// Why poll instead of having Python call us back? The ML service then needs to know nothing
// about this server or MongoDB - it stays a pure "compute" service that is easy to test alone.
import { config } from "./config.js";
import { HttpError } from "./errors.js";
import { mlRequest } from "./ml.js";
import { Run } from "./models/Run.js";

const watching = new Set(); // run ids with an active polling loop (never two loops for one run)
const MAX_NETWORK_FAILURES = 60; // ~1 minute of "ML service unreachable" before giving up

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Throws if the ML service refuses the job (unknown model, ...) so the upload request can fail cleanly.
export async function startVideoJob(run) {
  await mlRequest("/jobs/video", {
    method: "POST",
    json: {
      run_id: String(run._id),
      video_path: run.media_path,
      model: run.model,
      conf: run.conf,
      line: run.line,
    },
  });
  watchJob(String(run._id)); // not awaited: runs in the background
}

export async function watchJob(runId) {
  if (watching.has(runId)) return;
  watching.add(runId);
  let failures = 0;
  try {
    for (;;) {
      let job;
      try {
        job = await mlRequest(`/jobs/${runId}`, { timeoutMs: 10_000 });
        failures = 0;
      } catch (err) {
        if (err instanceof HttpError && err.status === 404) {
          // The ML service restarted (its jobs live in memory) or never had this job.
          await Run.updateOne({ _id: runId }, { status: "failed", error: "the ML service lost this job (was it restarted?)" });
          return;
        }
        if (++failures >= MAX_NETWORK_FAILURES) {
          await Run.updateOne({ _id: runId }, { status: "failed", error: "ML service unreachable" });
          return;
        }
        await sleep(config.jobPollMs);
        continue;
      }

      if (job.status === "done") {
        const r = job.result;
        await Run.updateOne(
          { _id: runId },
          {
            status: "done",
            progress: 1,
            width: job.width,
            height: job.height,
            fps: job.fps,
            frames: r.frames,
            avg_latency_ms: r.avg_latency_ms,
            summary: r.summary,
            tracks: r.tracks,
            events: r.events,
            samples: r.samples,
          },
        );
        await mlRequest(`/jobs/${runId}`, { method: "DELETE" }).catch(() => {});
        return;
      }
      if (job.status === "failed") {
        await Run.updateOne({ _id: runId }, { status: "failed", error: job.error });
        await mlRequest(`/jobs/${runId}`, { method: "DELETE" }).catch(() => {});
        return;
      }
      // queued / running: copy progress so the UI can show a progress bar
      await Run.updateOne(
        { _id: runId },
        { status: job.status, progress: job.progress, width: job.width, height: job.height, fps: job.fps },
      );
      await sleep(config.jobPollMs);
    }
  } catch (err) {
    console.error(`watching run ${runId} failed:`, err);
  } finally {
    watching.delete(runId);
  }
}

// At startup: runs that were queued/running when this server stopped get their polling loop back.
// If the ML service also restarted, the first poll answers 404 and the run is marked failed.
export async function recoverJobs() {
  const stuck = await Run.find({ status: { $in: ["queued", "running"] } }, { _id: 1 });
  for (const r of stuck) watchJob(String(r._id));
  return stuck.length;
}
