// Background processing of uploads. The upload request returns at once (202 Accepted);
// these jobs do the slow part by calling the ML service.
//
// Stage 1 - make it searchable (status: queued -> processing -> ready):
//     ML /index/documents: extract pages -> chunk -> embed -> add to FAISS + BM25
//     then the pages are saved in MongoDB (the document viewer reads them from there)
// Stage 2 - enrich it (enrich_status: pending -> running -> done):
//     ML /enrich: topic label (our classifier), entities (our NER model), summary (DistilBART)
//
// The two stages run on TWO separate queues, each handling one job at a time. Originally one worker
// did both; then a slow summary (its model was still downloading) left every newer upload stuck in
// "queued" - not even searchable - for minutes. Now a document becomes searchable as soon as the
// index queue reaches it, whatever the enrichment queue is doing. One job at a time per queue keeps
// GPU use predictable. At scale: a real queue (BullMQ + Redis, or Celery on the Python side) with N
// workers per stage - the document's status field in MongoDB stays the contract.
import { Document } from "./models/Document.js";
import { Page } from "./models/Page.js";
import { ml } from "./ml.js";

const log = (...a) => console.log("[jobs]", ...a);

// A tiny in-memory FIFO queue that runs one job at a time.
class Queue {
  constructor(name, handler) {
    this.name = name;
    this.handler = handler;
    this.items = [];
    this.running = false;
  }

  push(id) {
    this.items.push(String(id));
    if (!this.running) this.#drain(); // not awaited: the caller (an HTTP request) must not wait
  }

  async #drain() {
    this.running = true;
    while (this.items.length) {
      const id = this.items.shift();
      try {
        await this.handler(id);
      } catch (e) {
        // handlers record their own failures on the document; this only catches bugs (e.g. DB down)
        log(`${this.name} job ${id} crashed:`, e);
      }
    }
    this.running = false;
  }
}

async function indexJob(id) {
  const doc = await Document.findById(id);
  if (!doc) return; // deleted while waiting
  doc.status = "processing";
  await doc.save();
  try {
    const out = await ml.indexDocument(doc);
    await Page.deleteMany({ document_id: doc._id }); // a retried job must not duplicate pages
    await Page.insertMany(out.pages.map((p) => ({ document_id: doc._id, number: p.number, text: p.text })));
    doc.num_pages = out.pages.length;
    doc.num_chunks = out.num_chunks;
    doc.status = "ready";
    await doc.save();
    log(`indexed ${doc.filename}: ${doc.num_pages} pages, ${doc.num_chunks} chunks`);
  } catch (e) {
    log(`indexing ${doc.filename} failed:`, e.message);
    doc.status = "failed";
    doc.enrich_status = "skipped"; // never left "pending" forever
    doc.error = String(e.message).slice(0, 500);
    await doc.save();
    return;
  }
  enrichQueue.push(id);
}

async function enrichJob(id) {
  const doc = await Document.findById(id);
  if (!doc || doc.status !== "ready") return; // deleted in the meantime, or failed
  doc.enrich_status = "running";
  await doc.save();
  try {
    const pages = await Page.find({ document_id: doc._id }).sort({ number: 1 });
    const out = await ml.enrich(id, pages.map((p) => p.text).join("\n"));
    Object.assign(doc, {
      label: out.label,
      label_score: out.label_score,
      entities: out.entities,
      summary: out.summary,
      enrich_status: "done",
    });
    await doc.save();
  } catch (e) {
    // enrichment is optional: the document stays searchable
    log(`enrichment of ${doc.filename} failed:`, e.message);
    doc.enrich_status = "failed";
    doc.error = `enrichment: ${e.message}`.slice(0, 500);
    await doc.save();
  }
}

export const indexQueue = new Queue("index", indexJob);
export const enrichQueue = new Queue("enrich", enrichJob);

/** New upload: index it; when that succeeds the index job hands it to the enrichment queue. */
export const submit = (id) => indexQueue.push(id);

/**
 * Called at startup. The queues live in memory, so if the server stopped mid-job a document would
 * stay "queued"/"processing" (or enrichment "pending"/"running") forever. Reset those and run them again.
 * "pending" counts too: a document WAITING in the enrich queue when the server stopped is lost just
 * like one that was running. Re-indexing is safe: the ML service replaces a document's old chunks.
 */
export async function recover() {
  const stuck = await Document.find({
    $or: [
      { status: { $in: ["queued", "processing"] } },
      { status: "ready", enrich_status: { $in: ["pending", "running"] } },
    ],
  });
  for (const doc of stuck) {
    if (doc.status !== "ready") {
      // half-indexed: drop partial pages and start over
      await Page.deleteMany({ document_id: doc._id });
      doc.status = "queued";
      doc.enrich_status = "pending";
      await doc.save();
      indexQueue.push(doc.id);
    } else {
      doc.enrich_status = "pending";
      await doc.save();
      enrichQueue.push(doc.id);
    }
  }
  return stuck.length;
}
