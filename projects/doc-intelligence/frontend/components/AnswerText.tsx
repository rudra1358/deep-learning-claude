// Renders a generative answer, turning citation markers like "[2]" or "[1, 3]" into clickable chips.
// Clicking a chip scrolls to the matching source card (whose DOM id is `${idPrefix}-src-${n}`).

interface Props {
  text: string;
  idPrefix: string;
  sourceCount: number; // only numbers 1..sourceCount become chips; anything else stays plain text
}

// Matches [1], [2,3], [1, 2, 4]. The capture group holds the numbers.
const CITATION = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

export default function AnswerText({ text, idPrefix, sourceCount }: Props) {
  const parts: React.ReactNode[] = [];
  let last = 0;

  for (const match of text.matchAll(CITATION)) {
    const at = match.index ?? 0;
    parts.push(text.slice(last, at)); // plain text before the marker
    const numbers = match[1]!.split(",").map((s) => Number(s.trim()));
    for (const n of numbers) {
      if (n >= 1 && n <= sourceCount) {
        parts.push(
          <a
            key={`${at}-${n}`}
            href={`#${idPrefix}-src-${n}`}
            aria-label={`Go to source ${n}`}
            className="mx-0.5 inline-block rounded bg-accent px-1.5 align-baseline text-xs font-semibold text-accent-fg no-underline"
          >
            {n}
          </a>,
        );
      } else {
        parts.push(`[${n}]`);
      }
    }
    last = at + match[0].length;
  }
  parts.push(text.slice(last));

  return <p className="whitespace-pre-wrap break-words leading-relaxed">{parts}</p>;
}
