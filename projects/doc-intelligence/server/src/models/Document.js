// One uploaded file + its processing status and enrichments.
//
//   status         queued -> processing -> ready | failed     (stage 1: make it searchable)
//   enrich_status  pending -> running -> done | failed         (stage 2: label, entities, summary)
//                  "skipped" when stage 1 failed
//
// The extracted page texts live in their own collection (Page) because a long PDF can have
// hundreds of pages, and the document list should not load all of that text.
import mongoose from "mongoose";
import { jsonOptions } from "../db.js";

const documentSchema = new mongoose.Schema(
  {
    filename: { type: String, required: true },
    path: { type: String, required: true }, // where the upload is stored (not sent to the browser)
    status: { type: String, default: "queued", index: true },
    enrich_status: { type: String, default: "pending" },
    error: { type: String, default: null },
    num_pages: { type: Number, default: 0 },
    num_chunks: { type: Number, default: 0 },
    label: { type: String, default: null }, // topic from our fine-tuned classifier
    label_score: { type: Number, default: null },
    entities: { type: Array, default: null }, // [{text, type, count}] from our fine-tuned NER model
    summary: { type: String, default: null },
  },
  {
    timestamps: { createdAt: "created_at", updatedAt: false },
    toJSON: {
      ...jsonOptions,
      transform: (doc, ret) => {
        jsonOptions.transform(doc, ret);
        delete ret.path; // a server file path is nobody else's business
        return ret;
      },
    },
  },
);

export const Document = mongoose.model("Document", documentSchema);
