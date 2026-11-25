import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, errorMessage } from "./api";

/**
 * Run an async loader on mount and whenever `deps` change. Cancels the previous request
 * (AbortController) so a slow old response can never overwrite a newer one.
 * Returns {data, error, errorStatus (HTTP status, 0 = network error), loading, reload}.
 */
export function useAsync(loader, deps) {
  const [data, setData] = useState();
  const [error, setError] = useState(null);
  const [errorStatus, setErrorStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  const loaderRef = useRef(loader);
  loaderRef.current = loader;

  useEffect(() => {
    const ctrl = new AbortController();
    setLoading(true);
    setError(null);
    setErrorStatus(null);
    loaderRef
      .current(ctrl.signal)
      .then((d) => {
        if (!ctrl.signal.aborted) setData(d);
      })
      .catch((e) => {
        if (ctrl.signal.aborted) return;
        setError(errorMessage(e));
        setErrorStatus(e instanceof ApiError ? e.status : null);
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setLoading(false);
      });
    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, errorStatus, loading, reload };
}

/** Returns `value` only after it has stopped changing for `ms` milliseconds. */
export function useDebounced(value, ms) {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setV(value), ms);
    return () => clearTimeout(id);
  }, [value, ms]);
  return v;
}

/** Track an element's rendered size (CSS pixels). */
export function useElementSize() {
  const [size, setSize] = useState({ width: 0, height: 0 });
  const roRef = useRef(null);
  const ref = useCallback((el) => {
    roRef.current?.disconnect();
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const { width, height } = entry.contentRect;
      setSize((s) => (s.width === width && s.height === height ? s : { width, height }));
    });
    ro.observe(el);
    roRef.current = ro;
  }, []);
  return [ref, size];
}

/** Object URL for a File/Blob that is revoked automatically when it changes or unmounts. */
export function useObjectUrl(blob) {
  const [url, setUrl] = useState(null);
  useEffect(() => {
    if (!blob) {
      setUrl(null);
      return;
    }
    const u = URL.createObjectURL(blob);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [blob]);
  return url;
}

/** Set the browser tab title, e.g. "Image detection · Vision Analytics". */
export function usePageTitle(title) {
  useEffect(() => {
    document.title = title ? `${title} · Vision Analytics` : "Vision Analytics";
  }, [title]);
}
