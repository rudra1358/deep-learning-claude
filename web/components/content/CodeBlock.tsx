"use client";

import { Highlight, themes } from "prism-react-renderer";
import { Check, Copy, Play, TerminalSquare } from "lucide-react";
import { useState } from "react";
import { useTheme } from "@/components/layout/ThemeProvider";
import { runPython } from "@/lib/pyodide";

interface Props {
  code: string;
  lang: "python" | "bash" | "text";
  title?: string;
  output?: string;
  runnable?: boolean;
}

export function CodeBlock({ code, lang, title, output, runnable }: Props) {
  const { theme } = useTheme();
  const [copied, setCopied] = useState(false);
  const [showOut, setShowOut] = useState(false);
  const [live, setLive] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const src = code.replace(/^\n+|\s+$/g, "");

  async function copy() {
    await navigator.clipboard.writeText(src);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  async function run() {
    setRunning(true);
    setLive("Loading Python runtime (Pyodide + NumPy, first run only)…");
    const res = await runPython(src);
    setLive(res.error ? `${res.stdout}\n${res.error}` : res.stdout || "(no output)");
    setRunning(false);
  }

  return (
    <div className="my-5 overflow-hidden rounded-lg border border-border bg-code">
      <div className="flex items-center justify-between border-b border-border px-3 py-1.5 text-xs text-muted">
        <span className="font-mono">{title ?? lang}</span>
        <div className="flex items-center gap-1">
          {runnable && lang === "python" && (
            <button onClick={run} disabled={running} className="flex items-center gap-1 rounded px-2 py-1 hover:bg-surface-2 disabled:opacity-50" aria-label="Run code in browser">
              <Play size={13} /> {running ? "Running…" : "Run"}
            </button>
          )}
          {output && (
            <button onClick={() => setShowOut((s) => !s)} className="flex items-center gap-1 rounded px-2 py-1 hover:bg-surface-2" aria-expanded={showOut}>
              <TerminalSquare size={13} /> {showOut ? "Hide" : "Expected"} output
            </button>
          )}
          <button onClick={copy} className="flex items-center gap-1 rounded px-2 py-1 hover:bg-surface-2" aria-label="Copy code">
            {copied ? <Check size={13} /> : <Copy size={13} />} {copied ? "Copied" : "Copy"}
          </button>
        </div>
      </div>
      <Highlight code={src} language={lang === "text" ? "markup" : lang} theme={theme === "dark" ? themes.vsDark : themes.oneLight}>
        {({ tokens, getLineProps, getTokenProps }) => (
          <pre className="overflow-x-auto px-4 py-3 font-mono text-[0.82rem] leading-relaxed" style={{ background: "transparent" }}>
            {tokens.map((line, i) => (
              <div key={i} {...getLineProps({ line })}>
                {line.map((token, k) => (
                  <span key={k} {...getTokenProps({ token })} />
                ))}
              </div>
            ))}
          </pre>
        )}
      </Highlight>
      {showOut && output && (
        <div className="border-t border-border bg-surface px-4 py-3">
          <div className="mb-1 text-[0.7rem] uppercase tracking-wider text-faint">Expected output</div>
          <pre className="overflow-x-auto font-mono text-[0.8rem] text-muted">{output.replace(/^\n+|\s+$/g, "")}</pre>
        </div>
      )}
      {live !== null && (
        <div className="border-t border-border bg-surface px-4 py-3">
          <div className="mb-1 text-[0.7rem] uppercase tracking-wider text-faint">Live output (in-browser Python)</div>
          <pre className="max-h-80 overflow-auto font-mono text-[0.8rem] text-text">{live}</pre>
        </div>
      )}
    </div>
  );
}
