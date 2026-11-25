/**
 * Content schema for the platform.
 *
 * Every lesson MUST provide all 16 teaching sections (see SECTION_ORDER).
 * Text fields support a small inline-markdown dialect rendered by <Inline/>:
 *   **bold**   `code`   $inline LaTeX$   [link text](/learn/m04/backpropagation)
 * Write any string containing LaTeX backslashes with String.raw`...`.
 */

export type SimId =
  | "neuron"
  | "activations"
  | "network"
  | "gradient-descent"
  | "optimizers"
  | "loss-functions"
  | "convolution"
  | "cnn-pipeline"
  | "attention"
  | "fitting"
  | "matmul"
  | "chain-rule"
  | "positional-encoding"
  | "lstm"
  | "tokenizer"
  | "linear-transform"
  | "distributions"
  | "entropy"
  | "logistic-regression"
  | "approximator"
  | "gradient-flow"
  | "batchnorm"
  | "dropout"
  | "lr-schedule"
  | "regularization"
  | "metric-learning"
  | "receptive-field"
  | "conv-variants"
  | "rnn"
  | "bptt"
  | "gru"
  | "transformer"
  | "seq2seq-attention"
  | "detection"
  | "word-embeddings"
  | "autoencoder"
  | "vae"
  | "gan"
  | "diffusion"
  | "broadcasting"
  | "gnn"
  | "rl-gridworld"
  | "roc-pr"
  | "quantization"
  | "adversarial"
  | "decoding";

export interface QuizChoice {
  kind?: "choice";
  q: string;
  options: string[];
  answer: number; // index into options
  explain: string;
}

export interface QuizNumeric {
  kind: "numeric";
  q: string;
  answer: number;
  tolerance: number; // absolute tolerance
  explain: string;
}

export type QuizQuestion = QuizChoice | QuizNumeric;

export type Block =
  | { t: "p"; text: string }
  | { t: "h"; text: string }
  | { t: "math"; tex: string; caption?: string }
  | { t: "list"; items: string[]; ordered?: boolean }
  | { t: "code"; lang: "python" | "bash" | "text"; code: string; title?: string; output?: string; runnable?: boolean }
  | { t: "callout"; kind: "note" | "tip" | "warning" | "analogy" | "key"; title?: string; text: string }
  | { t: "table"; head: string[]; rows: string[][] }
  | { t: "sim"; id: SimId; caption?: string }
  | { t: "steps"; items: { title: string; text?: string; tex?: string }[] }
  | { t: "details"; summary: string; blocks: Block[] }
  | { t: "tabs"; tabs: { label: string; blocks: Block[] }[] }
  | { t: "qa"; level: "beginner" | "intermediate" | "advanced"; q: string; a: string }
  | { t: "quiz"; questions: QuizQuestion[] }
  | { t: "exercise"; title: string; prompt: string; hint?: string; solution: Block[] };

export const SECTION_ORDER = [
  ["intro", "Introduction"],
  ["prerequisites", "Prerequisites"],
  ["intuition", "Intuitive Explanation"],
  ["mechanism", "How It Actually Works"],
  ["math", "Mathematical Explanation"],
  ["visual", "Interactive Visualization"],
  ["scratch", "Code From Scratch (NumPy)"],
  ["framework", "Framework Implementation"],
  ["variations", "Types and Variations"],
  ["applications", "Practical Applications"],
  ["tradeoffs", "Advantages and Limitations"],
  ["mistakes", "Common Mistakes"],
  ["interview", "Interview Questions"],
  ["practice", "Knowledge Check"],
  ["summary", "Summary"],
  ["next", "Next Learning Path"],
] as const;

export type SectionKey = (typeof SECTION_ORDER)[number][0];

export type Level = "beginner" | "intermediate" | "advanced" | "research";

export interface Lesson {
  /** "m04/backpropagation" — must match a topic in curriculum.ts */
  id: string;
  title: string;
  summary: string;
  minutes: number;
  level: Level;
  /** lesson/topic ids, e.g. ["m02/derivatives-chain-rule"] */
  prerequisites: string[];
  /** ids of the next recommended topics */
  next: string[];
  /** glossary term keys introduced here */
  terms?: string[];
  sections: Record<SectionKey, Block[]>;
}

export interface Topic {
  slug: string;
  title: string;
  summary: string;
  /** Subtopics covered by this lesson (used for search + syllabus view) */
  subtopics: string[];
  level: Level;
}

export interface Module {
  id: string; // "m04"
  number: number;
  slug: string;
  title: string;
  tagline: string;
  phase: Phase;
  topics: Topic[];
}

export type Phase =
  | "Foundations"
  | "Neural Networks"
  | "Core Architectures"
  | "Applied Deep Learning"
  | "Generative AI"
  | "Advanced & Research"
  | "Engineering & Deployment";

export interface GlossaryTerm {
  key: string;
  term: string;
  definition: string;
  tex?: string;
  lesson?: string; // topic id
}
