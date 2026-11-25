// A "run" = one analysed image, uploaded video or live webcam session.
//
// In SQL this was 5 tables (runs, detections, tracks, events, count_samples) joined by run_id.
// In MongoDB the child rows are EMBEDDED arrays inside the run document, because they are always
// read and deleted together with their run. One document per run = one read, no joins.
// (Limit to know for interviews: a MongoDB document is max 16 MB. A 10-minute video produces a few
// thousand small rows here - far below that. The per-FRAME data is much bigger, so it stays on disk
// in data/runs/<id>/frames.json instead of in the database.)
import mongoose from "mongoose";

const { Schema } = mongoose;

const DetectionSchema = new Schema(
  {
    label: String,
    score: Number,
    box: [Number], // [x1, y1, x2, y2] in image pixels
  },
  { _id: false },
);

const TrackSchema = new Schema(
  {
    track_id: Number,
    label: String,
    first_frame: Number,
    last_frame: Number,
    max_score: Number,
  },
  { _id: false },
);

const EventSchema = new Schema(
  {
    track_id: Number,
    label: String,
    direction: { type: String, enum: ["in", "out"] },
    t_seconds: Number,
  },
  { _id: false },
);

const SampleSchema = new Schema(
  {
    t_seconds: Number, // second of the video
    label: String,
    count: Number,     // peak number of objects of this label visible during that second
  },
  { _id: false },
);

const RunSchema = new Schema(
  {
    kind: { type: String, enum: ["image", "video", "stream"], required: true, index: true },
    model: { type: String, required: true },
    source_name: { type: String, default: "" },
    media_path: { type: String, default: null }, // stored upload (videos only)
    status: { type: String, enum: ["queued", "running", "done", "failed"], default: "done", index: true },
    progress: { type: Number, default: 1 },
    error: { type: String, default: null },
    width: { type: Number, default: 0 },
    height: { type: Number, default: 0 },
    fps: { type: Number, default: 0 },
    frames: { type: Number, default: 0 },
    avg_latency_ms: { type: Number, default: 0 },
    conf: { type: Number, default: 0.25 },
    line: { type: [Number], default: null },     // counting line [x1, y1, x2, y2], normalised 0..1
    summary: { type: Schema.Types.Mixed, default: {} },
    detections: { type: [DetectionSchema], default: [] }, // image runs
    tracks: { type: [TrackSchema], default: [] },         // video / stream runs
    events: { type: [EventSchema], default: [] },         // line crossings
    samples: { type: [SampleSchema], default: [] },       // counts over time
  },
  {
    timestamps: { createdAt: "created_at", updatedAt: false },
    minimize: false, // keep empty objects like summary: {}
    toJSON: {
      // The API returns "id" (a string) instead of Mongo's "_id", and never the big embedded arrays -
      // those have their own endpoints (/detections, /events, /timeseries).
      transform(_doc, ret) {
        ret.id = String(ret._id);
        delete ret._id;
        delete ret.__v;
        delete ret.media_path;
        delete ret.detections;
        delete ret.tracks;
        delete ret.events;
        delete ret.samples;
        return ret;
      },
    },
  },
);

// Indexes for the history page: newest first, and "runs that contain a <label>".
RunSchema.index({ created_at: -1, _id: -1 });
RunSchema.index({ "detections.label": 1 });
RunSchema.index({ "tracks.label": 1 });

// Fields the run list does NOT need (keeps the history page fast).
export const HEAVY_FIELDS = "-detections -tracks -events -samples";

export const Run = mongoose.model("Run", RunSchema);
