// A conversation. Messages are EMBEDDED in the session document (not a separate collection):
// they are always read together with their session and never on their own - the typical case
// where MongoDB embedding beats a SQL-style join.
import mongoose from "mongoose";
import { jsonOptions } from "../db.js";

const messageSchema = new mongoose.Schema(
  {
    role: { type: String, enum: ["user", "assistant"], required: true },
    content: { type: String, required: true },
    sources: { type: Array, default: null }, // the chunks an answer was based on (null for user messages)
    meta: { type: mongoose.Schema.Types.Mixed, default: null }, // mode, found, cited, highlight, timing, ...
  },
  { timestamps: { createdAt: "created_at", updatedAt: false }, toJSON: jsonOptions },
);

const chatSessionSchema = new mongoose.Schema(
  {
    title: { type: String, default: "New chat" },
    messages: { type: [messageSchema], default: [] },
  },
  { timestamps: { createdAt: "created_at", updatedAt: false }, toJSON: jsonOptions },
);

export const ChatSession = mongoose.model("ChatSession", chatSessionSchema);
