import { useEffect, useState } from "react";
import { getHealth } from "../lib/api.js";

// Small status pill in the nav bar. Re-checks every 15 s so it turns green by itself
// once everything finishes starting (the ML service loading its models can take a while).
// The server reports its two dependencies separately, so the pill can say WHICH one is down.
export default function HealthBadge() {
  const [health, setHealth] = useState(null);
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
      <span role="status" className="text-xs text-bad" title="No response from the Express server">
        ● server unreachable
      </span>
    );
  }
  if (!health) return <span className="text-xs text-muted">● checking API…</span>;
  if (health.ml !== "ok" || health.mongo !== "ok") {
    const parts = [health.ml !== "ok" && "ML service", health.mongo !== "ok" && "MongoDB"].filter(Boolean);
    return (
      <span role="status" className="text-xs text-bad" title={health.ml_error ?? ""}>
        ● {parts.join(" + ")} down
      </span>
    );
  }

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
