// Search, question answering and chat history.  Mounted at /api.
//
// Search and answering are model work, so they are forwarded to the ML service. The server's own
// job is what the ML service does not know about: chat sessions and their messages (MongoDB).
import { Router } from "express";
import { isValidId } from "../db.js";
import { HttpError, ml } from "../ml.js";
import { ChatSession } from "../models/ChatSession.js";

const router = Router();

const SEARCH_MODES = ["hybrid", "dense", "bm25"];
const ASK_MODES = ["extractive", "generative"];

// Check the inputs here too, so a bad request gets a clear message without a trip to Python.
function text(v, name) {
  if (typeof v !== "string" || !v.trim() || v.length > 1000) throw new HttpError(422, `${name}: 1-1000 characters`);
  return v;
}
function documentIds(v) {
  if (v == null) return null;
  if (!Array.isArray(v) || !v.every(isValidId)) throw new HttpError(422, "document_ids: a list of document ids");
  return v.length ? v.map(String) : null;
}

router.post("/search", async (req, res) => {
  const { query, mode = "hybrid", rerank = true, top_k = 5, document_ids } = req.body ?? {};
  if (!SEARCH_MODES.includes(mode)) throw new HttpError(422, `mode: one of ${SEARCH_MODES.join(", ")}`);
  if (!Number.isInteger(top_k) || top_k < 1 || top_k > 50) throw new HttpError(422, "top_k: 1-50");
  res.json(
    await ml.search({ query: text(query, "query"), mode, rerank: Boolean(rerank), top_k, document_ids: documentIds(document_ids) }),
  );
});

router.post("/ask", async (req, res) => {
  const { question, mode = "extractive", document_ids, session_id = null } = req.body ?? {};
  text(question, "question");
  if (!ASK_MODES.includes(mode)) throw new HttpError(422, `mode: one of ${ASK_MODES.join(", ")}`);

  let session;
  if (session_id != null) {
    session = isValidId(session_id) ? await ChatSession.findById(session_id) : null;
    if (!session) throw new HttpError(404, "chat session not found");
  } else {
    session = new ChatSession({ title: question.slice(0, 80) }); // saved only if answering works
  }
  // Earlier turns go to the ML service: the generative model uses them for follow-up questions.
  const history = session.messages.map((m) => ({ role: m.role, content: m.content }));
  const result = await ml.ask({ question, mode, document_ids: documentIds(document_ids), history });

  const metaKeys = ["mode", "found", "cited", "highlight", "timing", "confidence", "suggestion", "gated"];
  session.messages.push(
    { role: "user", content: question },
    {
      role: "assistant",
      content: result.answer || "No answer found in the documents.",
      sources: result.sources,
      meta: Object.fromEntries(metaKeys.map((k) => [k, result[k] ?? null])),
    },
  );
  await session.save();
  res.json({ session_id: session.id, ...result });
});

router.get("/chats", async (_req, res) => {
  const sessions = await ChatSession.find({}, { messages: 0 }).sort({ created_at: -1, _id: -1 });
  res.json(sessions.map((s) => ({ id: s.id, title: s.title, created_at: s.created_at })));
});

async function findSession(id) {
  const s = isValidId(id) ? await ChatSession.findById(id) : null;
  if (!s) throw new HttpError(404, "chat session not found");
  return s;
}

router.get("/chats/:id", async (req, res) => {
  res.json(await findSession(req.params.id));
});

router.delete("/chats/:id", async (req, res) => {
  await (await findSession(req.params.id)).deleteOne();
  res.status(204).end();
});

export default router;
