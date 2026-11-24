import katex from "katex";
import Link from "next/link";
import { Fragment, type ReactNode } from "react";

/**
 * Renders the inline-markdown dialect used throughout the content:
 *   **bold**, `code`, $latex$, [text](href)
 * Kept deliberately tiny — content authors don't need full Markdown.
 */
const TOKEN = /(`[^`]+`)|(\$[^$]+\$)|(\*\*[^*]+\*\*)|(\*[^*\s][^*]*?\*)|(\[[^\]]+\]\([^)\s]+\))/g;

export function renderTex(tex: string, displayMode = false) {
  return katex.renderToString(tex, { displayMode, throwOnError: false, strict: "ignore" });
}

export function Inline({ text }: { text: string }) {
  const out: ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  TOKEN.lastIndex = 0;
  let i = 0;
  while ((m = TOKEN.exec(text))) {
    if (m.index > last) out.push(<Fragment key={i++}>{text.slice(last, m.index)}</Fragment>);
    const tok = m[0];
    if (tok.startsWith("`")) {
      out.push(
        <code key={i++} className="inline">
          {tok.slice(1, -1)}
        </code>,
      );
    } else if (tok.startsWith("$")) {
      const tex = tok.slice(1, -1);
      // KaTeX can only break inline math at top-level operators; long expressions get a
      // scrollable box so they never widen the page on phones. Short ones stay plain inline.
      out.push(<span key={i++} className={tex.length > 40 ? "katex-inline-long" : undefined} dangerouslySetInnerHTML={{ __html: renderTex(tex) }} />);
    } else if (tok.startsWith("**")) {
      out.push(
        <strong key={i++}>
          <Inline text={tok.slice(2, -2)} />
        </strong>,
      );
    } else if (tok.startsWith("*")) {
      out.push(
        <em key={i++}>
          <Inline text={tok.slice(1, -1)} />
        </em>,
      );
    } else {
      const lm =/^\[([^\]]+)\]\(([^)]+)\)$/.exec(tok)!;
      const href = lm[2];
      out.push(
        href.startsWith("/") ? (
          <Link key={i++} href={href}>
            {lm[1]}
          </Link>
        ) : (
          <a key={i++} href={href} target="_blank" rel="noreferrer">
            {lm[1]}
          </a>
        ),
      );
    }
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(<Fragment key={i++}>{text.slice(last)}</Fragment>);
  return <>{out}</>;
}

export function MathBlock({ tex, caption }: { tex: string; caption?: string }) {
  return (
    <figure className="my-5 min-w-0">
      <div className="overflow-x-auto rounded-lg border border-border bg-surface px-4 py-3" dangerouslySetInnerHTML={{ __html: renderTex(tex, true) }} />
      {caption && (
        <figcaption className="mt-1.5 text-center text-sm text-muted">
          <Inline text={caption} />
        </figcaption>
      )}
    </figure>
  );
}
