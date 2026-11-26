/**
 * Pure TypeScript inference for the trained GRU encoder–decoder with additive (Bahdanau) attention
 * exported by web/scripts/train-seq2seq-attention.py to /models/seq2seq-attention.json.
 *
 * Task: reverse a string of lowercase letters. Architecture (PyTorch conventions):
 *   x_j = E_src[token_j]                                   (16)
 *   →h_j = GRU_f(x_j, →h_{j−1}),  ←h_j = GRU_b(x_j, ←h_{j+1})   (24 each)
 *   h_j = [→h_j ; ←h_j]                                    (48, the "annotations")
 *   s_0 = tanh(W_init ←h_1 + b_init)                        (48)
 *   e_ij = vᵀ tanh(W_s s_{i−1} + W_h h_j)                  (attention dim 32)
 *   α_ij = softmax_j(e_ij),   c_i = Σ_j α_ij h_j
 *   s_i = GRUCell([E_tgt[y_{i−1}] ; c_i], s_{i−1})
 *   p_i = softmax(W_o [s_i ; c_i] + b_o)                    (29 symbols), y_i = argmax p_i
 *
 * GRU (PyTorch gate order r, z, n):
 *   r = σ(W_ir x + b_ir + W_hr h + b_hr), z = σ(W_iz x + b_iz + W_hz h + b_hz),
 *   n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn)),  h' = (1 − z) ⊙ n + z ⊙ h
 *
 * No React and no imports, so it can be unit-tested directly in Node.
 */

export type Mat = number[][];

export interface TensorJson {
  shape: number[];
  b64: string;
}

export interface Seq2SeqJson {
  task: string;
  config: { emb: number; encHidden: number; decHidden: number; att: number; vocabSize: number; pad: number; sos: number; eos: number; minLen: number; maxLen: number; maxDecodeSteps: number };
  vocab: string[];
  params: { attention: number; noAttention: number };
  accuracyByLength: { lengths: number[]; samplesPerLength: number; attention: number[]; noAttention: number[] };
  weights: Record<string, TensorJson>;
  verification: VerificationTrace[];
}

export interface VerificationTrace {
  input: string;
  H: Mat;
  s0: number[];
  output: string;
  /** s_{i−1} is not stored: it is s0 for step 1 and the previous step's s afterwards. */
  steps: { e: number[]; alpha: number[]; context: number[]; s: number[]; probs: number[]; token: number }[];
}

interface GruWeights {
  Wih: Mat; // (3H, in)
  Whh: Mat; // (3H, H)
  bih: number[];
  bhh: number[];
}

export interface Seq2SeqModel {
  cfg: Seq2SeqJson["config"];
  vocab: string[];
  stoi: Map<string, number>;
  srcEmb: Mat;
  encF: GruWeights;
  encB: GruWeights;
  initW: Mat;
  initB: number[];
  Ws: Mat; // (att, decHidden)
  Wh: Mat; // (att, 2·encHidden)
  v: number[]; // (att)
  tgtEmb: Mat;
  dec: GruWeights;
  outW: Mat; // (vocab, decHidden + 2·encHidden)
  outB: number[];
}

/* ------------------------------------------------------------------ decoding the JSON */

function b64ToFloats(b64: string): number[] {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return Array.from(new Float32Array(bytes.buffer)); // little-endian float32, as written by numpy "<f4"
}

function toMat(t: TensorJson): Mat {
  const flat = b64ToFloats(t.b64);
  const [r, c] = t.shape.length === 2 ? t.shape : [1, t.shape[0]];
  return Array.from({ length: r }, (_, i) => flat.slice(i * c, (i + 1) * c));
}
const toVec = (t: TensorJson) => b64ToFloats(t.b64);

export function loadModel(json: Seq2SeqJson): Seq2SeqModel {
  const w = json.weights;
  const gru = (p: string): GruWeights => ({ Wih: toMat(w[p + "Wih"]), Whh: toMat(w[p + "Whh"]), bih: toVec(w[p + "bih"]), bhh: toVec(w[p + "bhh"]) });
  return {
    cfg: json.config,
    vocab: json.vocab,
    stoi: new Map(json.vocab.map((s, i) => [s, i])),
    srcEmb: toMat(w.src_emb),
    encF: gru("enc_f_"),
    encB: gru("enc_b_"),
    initW: toMat(w.init_W),
    initB: toVec(w.init_b),
    Ws: toMat(w.W_s),
    Wh: toMat(w.W_h),
    v: toVec(w.v),
    tgtEmb: toMat(w.tgt_emb),
    dec: gru("dec_"),
    outW: toMat(w.out_W),
    outB: toVec(w.out_b),
  };
}

/* ------------------------------------------------------------------ small linear algebra */

export function matVec(M: Mat, x: number[], b?: number[]): number[] {
  return M.map((row, i) => {
    let s = b ? b[i] : 0;
    for (let k = 0; k < row.length; k++) s += row[k] * x[k];
    return s;
  });
}
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export function softmax(v: number[]): number[] {
  const m = Math.max(...v);
  const e = v.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

/** One PyTorch-convention GRU step. Returns the new state and its gates (for display). */
export function gruStep(g: GruWeights, x: number[], h: number[]) {
  const H = h.length;
  const gi = matVec(g.Wih, x, g.bih);
  const gh = matVec(g.Whh, h, g.bhh);
  const r = new Array<number>(H), z = new Array<number>(H), n = new Array<number>(H), out = new Array<number>(H);
  for (let k = 0; k < H; k++) {
    r[k] = sigmoid(gi[k] + gh[k]);
    z[k] = sigmoid(gi[H + k] + gh[H + k]);
    n[k] = Math.tanh(gi[2 * H + k] + r[k] * gh[2 * H + k]);
    out[k] = (1 - z[k]) * n[k] + z[k] * h[k];
  }
  return { h: out, r, z, n };
}

/* ------------------------------------------------------------------ input handling */

export function validateInput(model: Seq2SeqModel, text: string): string | null {
  if (text.length === 0) return "Type at least one letter.";
  if (text.length > model.cfg.maxLen) return `Use at most ${model.cfg.maxLen} letters (the model learned on lengths ${model.cfg.minLen}–${model.cfg.maxLen}).`;
  const bad = [...text].filter((c) => !model.stoi.has(c) || (model.stoi.get(c) ?? 0) < 3);
  if (bad.length) return `The model only knows lowercase letters a–z (found "${[...new Set(bad)].join("")}").`;
  return null;
}

/* ------------------------------------------------------------------ encoder */

export interface EncoderResult {
  ids: number[];
  fwd: Mat; // →h_j
  bwd: Mat; // ←h_j
  H: Mat; // annotations [→h_j ; ←h_j]
  keys: Mat; // W_h h_j, precomputed once per input
  s0: number[];
}

export function encodeInput(model: Seq2SeqModel, text: string): EncoderResult {
  const ids = [...text].map((c) => model.stoi.get(c) ?? 0);
  const T = ids.length, Hn = model.cfg.encHidden;
  const xs = ids.map((id) => model.srcEmb[id]);
  const fwd: Mat = new Array(T), bwd: Mat = new Array(T);
  let h = new Array<number>(Hn).fill(0);
  for (let j = 0; j < T; j++) fwd[j] = h = gruStep(model.encF, xs[j], h).h;
  h = new Array<number>(Hn).fill(0);
  for (let j = T - 1; j >= 0; j--) bwd[j] = h = gruStep(model.encB, xs[j], h).h;
  const H = fwd.map((f, j) => [...f, ...bwd[j]]);
  const keys = H.map((hj) => matVec(model.Wh, hj));
  const s0 = matVec(model.initW, bwd[0], model.initB).map(Math.tanh);
  return { ids, fwd, bwd, H, keys, s0 };
}

/* ------------------------------------------------------------------ one decoder step */

export interface DecodeStep {
  i: number; // 1-based output position
  yPrev: number; // previous token id (SOS at i = 1)
  sPrev: number[]; // s_{i−1}
  query: number[]; // W_s s_{i−1}
  e: number[]; // additive scores e_ij for every input position j
  alpha: number[]; // softmax weights
  context: number[]; // c_i
  s: number[]; // s_i
  logits: number[];
  probs: number[];
  top: { id: number; sym: string; p: number }[]; // top-5
  token: number; // argmax
  sym: string;
}

export function decodeStep(model: Seq2SeqModel, enc: EncoderResult, sPrev: number[], yPrev: number, i: number): DecodeStep {
  const query = matVec(model.Ws, sPrev);
  const e = enc.keys.map((k) => {
    let s = 0;
    for (let a = 0; a < query.length; a++) s += model.v[a] * Math.tanh(query[a] + k[a]);
    return s;
  });
  const alpha = softmax(e);
  const D = enc.H[0].length;
  const context = new Array<number>(D).fill(0);
  enc.H.forEach((hj, j) => {
    for (let d = 0; d < D; d++) context[d] += alpha[j] * hj[d];
  });
  const s = gruStep(model.dec, [...model.tgtEmb[yPrev], ...context], sPrev).h;
  const logits = matVec(model.outW, [...s, ...context], model.outB);
  const probs = softmax(logits);
  const order = probs.map((p, id) => ({ id, p })).sort((a, b) => b.p - a.p);
  const top = order.slice(0, 5).map(({ id, p }) => ({ id, p, sym: model.vocab[id] }));
  const token = order[0].id;
  return { i, yPrev, sPrev, query, e, alpha, context, s, logits, probs, top, token, sym: model.vocab[token] };
}

/* ------------------------------------------------------------------ greedy decoding */

export interface Seq2SeqTrace {
  input: string;
  enc: EncoderResult;
  steps: DecodeStep[];
  output: string; // decoded letters (without <eos>)
  ended: boolean; // true if <eos> was produced
  correct: boolean; // output equals the reversed input
}

export function greedyDecode(model: Seq2SeqModel, text: string, maxSteps = model.cfg.maxDecodeSteps): Seq2SeqTrace {
  const enc = encodeInput(model, text);
  const steps: DecodeStep[] = [];
  let s = enc.s0, y = model.cfg.sos, ended = false;
  for (let i = 1; i <= maxSteps; i++) {
    const st = decodeStep(model, enc, s, y, i);
    steps.push(st);
    s = st.s;
    y = st.token;
    if (y === model.cfg.eos) {
      ended = true;
      break;
    }
  }
  const output = steps
    .filter((st) => st.token !== model.cfg.eos)
    .map((st) => (st.token >= 3 ? st.sym : "?"))
    .join("");
  return { input: text, enc, steps, output, ended, correct: ended && output === [...text].reverse().join("") };
}

/** Max |difference| between this engine and a PyTorch verification trace (stored to 5 decimals). */
export function compareWithReference(model: Seq2SeqModel, ref: VerificationTrace): { maxDiff: number; sameTokens: boolean } {
  const tr = greedyDecode(model, ref.input);
  let maxDiff = 0;
  const upd = (a: number[], b: number[]) => a.forEach((x, k) => (maxDiff = Math.max(maxDiff, Math.abs(x - b[k]))));
  tr.enc.H.forEach((row, j) => upd(row, ref.H[j]));
  upd(tr.enc.s0, ref.s0);
  const n = Math.min(tr.steps.length, ref.steps.length);
  for (let k = 0; k < n; k++) {
    const a = tr.steps[k], b = ref.steps[k];
    upd(a.e, b.e);
    upd(a.alpha, b.alpha);
    upd(a.context, b.context);
    upd(a.s, b.s);
    upd(a.probs, b.probs);
  }
  const sameTokens = tr.steps.length === ref.steps.length && tr.steps.every((st, k) => st.token === ref.steps[k].token);
  return { maxDiff, sameTokens };
}
