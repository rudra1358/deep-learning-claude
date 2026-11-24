"use client";

import { useSyncExternalStore } from "react";

/**
 * Learner progress store.
 *
 * Source of truth is localStorage (works offline, no account needed).
 * If NEXT_PUBLIC_API_URL is set, every change is also pushed to the FastAPI
 * backend (backend/app) keyed by an anonymous learner id, and pulled on load,
 * so progress survives across devices once accounts are added.
 */

export interface ProgressState {
  learnerId: string;
  completed: Record<string, number>; // topicId -> completed timestamp
  quizzes: Record<string, { score: number; total: number; at: number }>; // quizKey -> best
  bookmarks: string[];
  projects: Record<string, string[]>; // projectId -> completed milestone ids
  activeDays: string[]; // YYYY-MM-DD
  lastVisited?: string;
}

const KEY = "dlu-progress-v1";
const API = process.env.NEXT_PUBLIC_API_URL;

function blank(): ProgressState {
  return {
    learnerId: typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : String(Date.now()),
    completed: {},
    quizzes: {},
    bookmarks: [],
    projects: {},
    activeDays: [],
  };
}

let state: ProgressState | null = null;
const listeners = new Set<() => void>();
const SERVER_SNAPSHOT = blank();

function load(): ProgressState {
  if (state) return state;
  if (typeof window === "undefined") return SERVER_SNAPSHOT;
  try {
    const raw = localStorage.getItem(KEY);
    state = raw ? { ...blank(), ...JSON.parse(raw) } : blank();
  } catch {
    state = blank();
  }
  void pull();
  return state!;
}

async function pull() {
  if (!API || !state) return;
  try {
    const r = await fetch(`${API}/api/progress/${state.learnerId}`);
    if (!r.ok) return;
    const remote = (await r.json()) as Partial<ProgressState>;
    state = mergeStates(state, remote);
    persist(false);
  } catch {
    /* offline: local copy remains the source of truth */
  }
}

/** Union merge — never lose a completion or a better quiz score. */
export function mergeStates(a: ProgressState, b: Partial<ProgressState>): ProgressState {
  const quizzes = { ...a.quizzes };
  for (const [k, v] of Object.entries(b.quizzes ?? {})) {
    if (!quizzes[k] || v.score / v.total > quizzes[k].score / quizzes[k].total) quizzes[k] = v;
  }
  const projects = { ...a.projects };
  for (const [k, v] of Object.entries(b.projects ?? {})) projects[k] = Array.from(new Set([...(projects[k] ?? []), ...v]));
  return {
    ...a,
    completed: { ...(b.completed ?? {}), ...a.completed },
    quizzes,
    bookmarks: Array.from(new Set([...a.bookmarks, ...(b.bookmarks ?? [])])),
    projects,
    activeDays: Array.from(new Set([...a.activeDays, ...(b.activeDays ?? [])])).sort(),
  };
}

function persist(push = true) {
  if (!state) return;
  try {
    localStorage.setItem(KEY, JSON.stringify(state));
  } catch {}
  listeners.forEach((l) => l());
  if (push && API) {
    fetch(`${API}/api/progress/${state.learnerId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state),
    }).catch(() => {});
  }
}

function today() {
  return new Date().toISOString().slice(0, 10);
}

function update(fn: (s: ProgressState) => ProgressState) {
  const cur = load();
  const next = fn(cur);
  const d = today();
  state = next.activeDays.includes(d) ? next : { ...next, activeDays: [...next.activeDays, d] };
  persist();
}

export const progress = {
  toggleComplete(topicId: string) {
    update((s) => {
      const completed = { ...s.completed };
      if (completed[topicId]) delete completed[topicId];
      else completed[topicId] = Date.now();
      return { ...s, completed };
    });
  },
  recordQuiz(key: string, score: number, total: number) {
    update((s) => {
      const prev = s.quizzes[key];
      if (prev && prev.score / prev.total >= score / total) return s;
      return { ...s, quizzes: { ...s.quizzes, [key]: { score, total, at: Date.now() } } };
    });
  },
  toggleBookmark(topicId: string) {
    update((s) => ({
      ...s,
      bookmarks: s.bookmarks.includes(topicId) ? s.bookmarks.filter((b) => b !== topicId) : [...s.bookmarks, topicId],
    }));
  },
  toggleMilestone(projectId: string, milestone: string) {
    update((s) => {
      const cur = s.projects[projectId] ?? [];
      const next = cur.includes(milestone) ? cur.filter((m) => m !== milestone) : [...cur, milestone];
      return { ...s, projects: { ...s.projects, [projectId]: next } };
    });
  },
  visit(topicId: string) {
    update((s) => ({ ...s, lastVisited: topicId }));
  },
  reset() {
    state = blank();
    persist();
  },
};

/** Consecutive-day streak ending today (or yesterday). */
export function streak(days: string[]): number {
  const set = new Set(days);
  let n = 0;
  const d = new Date();
  if (!set.has(d.toISOString().slice(0, 10))) d.setDate(d.getDate() - 1);
  while (set.has(d.toISOString().slice(0, 10))) {
    n++;
    d.setDate(d.getDate() - 1);
  }
  return n;
}

export function useProgress(): ProgressState {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    load,
    () => SERVER_SNAPSHOT,
  );
}
