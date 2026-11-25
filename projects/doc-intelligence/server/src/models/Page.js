// Extracted text of one page of a document (the document viewer shows these).
import mongoose from "mongoose";
import { jsonOptions } from "../db.js";

const pageSchema = new mongoose.Schema(
  {
    document_id: { type: mongoose.Schema.Types.ObjectId, ref: "Document", required: true },
    number: { type: Number, required: true }, // 1-based
    text: { type: String, default: "" },
  },
  { toJSON: jsonOptions },
);
// The viewer asks for "all pages of document X in order" - this index answers exactly that.
pageSchema.index({ document_id: 1, number: 1 });

export const Page = mongoose.model("Page", pageSchema);
