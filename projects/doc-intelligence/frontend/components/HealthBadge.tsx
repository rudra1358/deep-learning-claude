"use client";

import { useEffect, useState } from "react";
import { API_URL, getHealth, type Health } from "@/lib/api";

// Small status pill in the nav bar. Re-checks every 15 s so it turns green by itself
// once the backend finishes starting (loading the models can take a while).
export default function HealthBadge() {
  const [health, setHealth] = useState<Health | null>(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    let cancelled = false; // ignore answers that arrive after the component is gone

    async function check() {
      try {
        const h = await getHealth();
        if (!cancelled) {
          setHealth(h);
          setDown(false);
        }
      } catch {
        if (!cancelled) setDown(true);
      }
    }

    check();
    const timer = setInterval(check, 15_000);
    return () => {
      cancelled = true;
      clearInterval(timer); // cleanup: stop polling when the component unmounts
    };
  }, []);

  if (down) {
    return (
      <span role="status" className="text-xs text-bad" title={`No response from ${API_URL}`}>
        ● API unreachable
      </span>
    );
  }
  if (!health) return <span className="text-xs text-muted">● checking API…</span>;

  return (
    <span
      role="status"
      className="text-xs text-muted"
      title={`Embedder: ${health.embedder}\nModels loaded: ${health.models_loaded.join(", ")}`}
    >
      <span className="text-ok">●</span> API ok · {health.device} · {health.chunks_indexed} chunks · generator{" "}
      {health.generator_enabled ? "on" : "off"}
    </span>
  );
}
