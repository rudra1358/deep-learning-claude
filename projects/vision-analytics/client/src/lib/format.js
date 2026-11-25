import { parseServerDate } from "./api";

export const fmtMs = (v, digits = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${v.toFixed(digits)} ms`;

export const fmtNum = (v, digits = 3) => (v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(digits));

export const fmtPct = (v, digits = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export function fmtDate(s) {
  const d = parseServerDate(s);
  return Number.isNaN(d.getTime()) ? s : d.toLocaleString();
}

export function fmtSeconds(s) {
  const m = Math.floor(s / 60);
  const sec = s - m * 60;
  return m > 0 ? `${m}:${sec.toFixed(1).padStart(4, "0")}` : `${sec.toFixed(1)} s`;
}

export const sumValues = (r) => (r ? Object.values(r).reduce((a, b) => a + b, 0) : 0);

/** Sort a {label: n} record into [label, n] pairs, largest first. */
export const sortedEntries = (r) => (r ? Object.entries(r).sort((a, b) => b[1] - a[1]) : []);

/** Run ids are 24-character MongoDB ObjectIds; the last 6 characters are enough to tell runs apart on screen. */
export const shortId = (id) => String(id ?? "").slice(-6);
