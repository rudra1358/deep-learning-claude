import { useEffect, useRef, useState } from "react";
import AssistantMessage, { turnFromAsk, turnFromHistory } from "../components/AssistantMessage.jsx";
import ChatSidebar from "../components/ChatSidebar.jsx";
import DocFilter from "../components/DocFilter.jsx";
import { ErrorBox, Loading } from "../components/ui.jsx";
import { ask, errorMessage, getChat, getHealth } from "../lib/api.js";

// The Ask page: a chat over your documents.
// Flow: type a question -> POST /api/ask -> show the answer + numbered sources.
// The server returns a session_id; sending it back with the next question continues the same chat
// (the server sends the earlier turns to the ML service, and the generative model sees them as history).
export default function AskPage() {
  const [turns, setTurns] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("extractive"); // "extractive" | "generative"
  const [docIds, setDocIds] = useState([]);
  const [asking, setAsking] = useState(false);
  const [loadingChat, setLoadingChat] = useState(false);
  const [error, setError] = useState("");
  const [historyVersion, setHistoryVersion] = useState(0); // bump to make the sidebar reload
  const [generatorOn, setGeneratorOn] = useState(true);
  const endRef = useRef(null);

  // Ask the server once whether the LLM is enabled, so we can disable the Generative option up front.
  useEffect(() => {
    getHealth()
      .then((h) => setGeneratorOn(h.generator_enabled ?? true))
      .catch(() => {}); // the nav badge already reports an unreachable API
  }, []);

  // Keep the newest message in view.
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, asking]);

  async function submit(e) {
    e?.preventDefault();
    const q = question.trim();
    if (!q || asking) return;
    setError("");
    setAsking(true);
    setQuestion("");
    setTurns((prev) => [...prev, { role: "user", text: q }]); // show the question immediately
    try {
      const res = await ask({
        question: q,
        mode,
        session_id: sessionId,
        document_ids: docIds.length ? docIds : null,
      });
      setSessionId(res.session_id);
      setTurns((prev) => [...prev, turnFromAsk(res)]);
      setHistoryVersion((v) => v + 1);
    } catch (err) {
      // Undo the optimistic question and put the text back so the user can retry.
      setTurns((prev) => prev.slice(0, -1));
      setQuestion(q);
      setError(errorMessage(err));
    } finally {
      setAsking(false);
    }
  }

  async function openChat(id) {
    setError("");
    setLoadingChat(true);
    try {
      const chat = await getChat(id);
      setTurns(chat.messages.map(turnFromHistory));
      setSessionId(chat.id);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoadingChat(false);
    }
  }

  function newChat() {
    setTurns([]);
    setSessionId(null);
    setError("");
  }

  return (
    <div className="flex flex-col gap-6 md:flex-row">
      <ChatSidebar
        activeId={sessionId}
        refreshKey={historyVersion}
        onSelect={openChat}
        onNew={newChat}
        onDeleted={(id) => id === sessionId && newChat()}
      />

      <section className="min-w-0 flex-1 space-y-4" aria-label="Conversation">
        {loadingChat && <Loading text="Loading conversation…" />}
        {!loadingChat && turns.length === 0 && (
          <div className="rounded border border-dashed border-border p-6 text-sm text-muted">
            <p className="mb-1 font-medium text-fg">Ask a question about your documents.</p>
            <p>
              <b>Extractive</b> returns an exact quote from a document (fast, never invents text).{" "}
              <b>Generative</b> lets a small LLM write an answer with [n] citations to the sources.
            </p>
          </div>
        )}

        {turns.map((t, i) =>
          t.role === "user" ? (
            <p key={i} className="ml-auto w-fit max-w-[85%] whitespace-pre-wrap break-words rounded bg-accent px-3 py-2 text-accent-fg">
              {t.text}
            </p>
          ) : (
            <AssistantMessage key={i} turn={t} idPrefix={`m${i}`} />
          ),
        )}
        {asking && <Loading text={mode === "generative" ? "Writing an answer… (the LLM can take a while)" : "Searching…"} />}
        {error && <ErrorBox message={error} />}
        <div ref={endRef} />

        <form onSubmit={submit} className="sticky bottom-0 space-y-2 rounded border border-border bg-card p-3 shadow-sm">
          <label htmlFor="question" className="sr-only">
            Your question
          </label>
          <textarea
            id="question"
            rows={2}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              // Enter sends, Shift+Enter makes a new line (like most chat apps).
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder="e.g. What is the notice period in the contract?"
            maxLength={1000}
            className="w-full resize-y rounded border border-border bg-bg p-2"
          />
          <div className="flex flex-wrap items-center gap-2">
            <fieldset className="flex overflow-hidden rounded border border-border text-sm">
              <legend className="sr-only">Answer mode</legend>
              {["extractive", "generative"].map((m) => (
                <label
                  key={m}
                  className={`cursor-pointer px-3 py-1.5 has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-inset has-[:focus-visible]:ring-accent ${mode === m ? "bg-accent text-accent-fg" : ""} ${
                    m === "generative" && !generatorOn ? "cursor-not-allowed opacity-50" : ""
                  }`}
                  title={m === "generative" && !generatorOn ? "The LLM is disabled on this server" : undefined}
                >
                  <input
                    type="radio"
                    name="mode"
                    value={m}
                    checked={mode === m}
                    disabled={m === "generative" && !generatorOn}
                    onChange={() => setMode(m)}
                    className="sr-only"
                  />
                  {m === "extractive" ? "Extractive" : "Generative"}
                </label>
              ))}
            </fieldset>
            <DocFilter selected={docIds} onChange={setDocIds} />
            <button
              type="submit"
              disabled={asking || !question.trim()}
              className="ml-auto rounded bg-accent px-4 py-1.5 text-sm font-medium text-accent-fg disabled:opacity-50"
            >
              {asking ? "Asking…" : "Ask"}
            </button>
          </div>
        </form>
      </section>
    </div>
  );
}
