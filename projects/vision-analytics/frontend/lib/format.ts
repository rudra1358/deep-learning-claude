import { parseServerDate } from "./api";

export const fmtMs = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${v.toFixed(digits)} ms`;

export const fmtNum = (v: number | null | undefined, digits = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(digits);

export const fmtPct = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export function fmtDate(s: string): string {
  const d = parseServerDate(s);
  return Number.isNaN(d.getTime()) ? s : d.toLocaleString();
}

export function fmtSeconds(s: number): string {
  const m = Math.floor(s / 60);
  const sec = s - m * 60;
  return m > 0 ? `${m}:${sec.toFixed(1).padStart(4, "0")}` : `${sec.toFixed(1)} s`;
}

export const sumValues = (r: Record<string, number> | undefined | null) =>
  r ? Object.values(r).reduce((a, b) => a + b, 0) : 0;

/** Sort a {label: n} record into [label, n] pairs, largest first. */
export const sortedEntries = (r: Record<string, number> | undefined | null): [string, number][] =>
  r ? Object.entries(r).sort((a, b) => b[1] - a[1]) : [];
