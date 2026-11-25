"use client";

import { useEffect, useState } from "react";
import { deleteChat, errorMessage, listChats, type SessionOut } from "@/lib/api";

interface Props {
  activeId: number | null;
  refreshKey: number; // the Ask page bumps this after each question so a new chat appears in the list
  onSelect: (id: number) => void;
  onNew: () => void;
  onDeleted: (id: number) => void;
}

// Left column of the Ask page: past conversations from GET /api/chats.
// On phones it is hidden behind a "History" button so the chat gets the full width.
export default function ChatSidebar({ activeId, refreshKey, onSelect, onNew, onDeleted }: Props) {
  const [chats, setChats] = useState<SessionOut[] | null>(null);
  const [error, setError] = useState("");
  const [open, setOpen] = useState(false); // only matters on small screens

  useEffect(() => {
    listChats()
      .then((c) => {
        setChats(c);
        setError("");
      })
      .catch((e) => setError(errorMessage(e)));
  }, [refreshKey]); // re-run whenever the parent says something changed

  async function remove(chat: SessionOut) {
    if (!window.confirm(`Delete the conversation "${chat.title}"?`)) return;
    try {
      await deleteChat(chat.id);
      setChats((prev) => (prev ?? []).filter((c) => c.id !== chat.id));
      onDeleted(chat.id);
    } catch (e) {
      setError(errorMessage(e));
    }
  }

  return (
    <aside className="md:w-64 md:shrink-0">
      <div className="mb-2 flex gap-2">
        <button type="button" onClick={onNew} className="flex-1 rounded bg-accent px-3 py-2 text-sm font-medium text-accent-fg">
          + New chat
        </button>
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          aria-controls="chat-history"
          className="rounded border border-border px-3 py-2 text-sm md:hidden"
        >
          History
        </button>
      </div>

      <nav id="chat-history" aria-label="Chat history" className={`${open ? "block" : "hidden"} md:block`}>
        {error && <p className="text-sm text-bad">{error}</p>}
        {!error && chats === null && <p className="text-sm text-muted">Loading chats…</p>}
        {chats?.length === 0 && <p className="text-sm text-muted">No conversations yet.</p>}
        <ul className="space-y-1">
          {chats?.map((c) => (
            <li key={c.id} className="group flex items-center gap-1">
              <button
                type="button"
                onClick={() => {
                  onSelect(c.id);
                  setOpen(false);
                }}
                aria-current={c.id === activeId ? "true" : undefined}
                className={`min-w-0 flex-1 truncate rounded px-2 py-1.5 text-left text-sm ${
                  c.id === activeId ? "bg-card font-medium" : "text-muted hover:bg-card"
                }`}
                title={c.title}
              >
                {c.title}
              </button>
              <button
                type="button"
                onClick={() => remove(c)}
                aria-label={`Delete conversation ${c.title}`}
                className="rounded px-2 py-1 text-muted hover:text-bad"
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      </nav>
    </aside>
  );
}
