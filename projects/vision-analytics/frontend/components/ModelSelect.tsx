"use client";

import { useEffect, useMemo } from "react";
import { getModels, type ModelInfo } from "@/lib/api";
import { useAsync } from "@/lib/hooks";

/** Loads GET /api/models once per mount. */
export function useModels() {
  return useAsync<ModelInfo[]>((signal) => getModels({ signal }), []);
}

/**
 * <select> of models from GET /api/models. Selects the first model automatically when
 * `value` is empty or no longer available.
 */
export function ModelSelect({
  id,
  label = "Model",
  value,
  onChange,
  disabled,
  models,
  exclude,
}: {
  id: string;
  label?: string;
  value: string;
  onChange: (name: string) => void;
  disabled?: boolean;
  /** result of useModels() in the page (shared when a page has several selects) */
  models: { data: ModelInfo[] | undefined; error: string | null; loading: boolean };
  exclude?: string;
}) {
  const src = models;
  const options = useMemo(() => (src.data ?? []).filter((m) => m.name !== exclude), [src.data, exclude]);

  useEffect(() => {
    const first = options[0];
    if (first && !options.some((m) => m.name === value)) onChange(first.name);
  }, [options, value, onChange]);

  return (
    <div>
      <label htmlFor={id} className="label">
        {label}
      </label>
      <select
        id={id}
        className="input"
        value={value}
        disabled={disabled || src.loading || options.length === 0}
        onChange={(e) => onChange(e.target.value)}
      >
        {src.loading && <option value="">Loading models…</option>}
        {!src.loading && options.length === 0 && <option value="">No models available</option>}
        {options.map((m) => (
          <option key={m.name} value={m.name}>
            {m.name}
            {m.loaded ? "" : " (not loaded yet)"}
          </option>
        ))}
      </select>
      {src.error && <p className="mt-1 text-xs text-danger">Could not load models: {src.error}</p>}
      {!src.loading && !src.error && options.length === 0 && (
        <p className="mt-1 text-xs text-muted">No .onnx models found on the server — export a model first.</p>
      )}
    </div>
  );
}
