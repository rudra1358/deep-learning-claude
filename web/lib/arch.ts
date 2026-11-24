/** Layer-spec shape & parameter calculator used by the Architecture Explorer. */

export type LayerSpec =
  | { type: "conv"; out: number; k: number; s?: number; p?: number; name?: string }
  | { type: "pool"; k: number; s?: number; mode?: "max" | "avg"; name?: string }
  | { type: "adaptivepool"; size: number; name?: string }
  | { type: "flatten"; name?: string }
  | { type: "dense"; out: number; name?: string }
  | { type: "relu" | "dropout"; name?: string };

export interface LayerRow {
  name: string;
  op: string;
  shape: string;
  params: number;
  formula: string;
}

type Shape = { c: number; h: number; w: number } | { f: number };

export function computeCnn(input: { c: number; h: number; w: number }, layers: LayerSpec[]): { rows: LayerRow[]; total: number; error?: string } {
  let shape: Shape = { ...input };
  const rows: LayerRow[] = [{ name: "input", op: "Input", shape: fmtShape(shape), params: 0, formula: "" }];
  let total = 0;
  for (const [i, L] of layers.entries()) {
    const name = L.name ?? `${L.type}${i + 1}`;
    if (L.type === "conv") {
      if (!("c" in shape)) return { rows, total, error: `${name}: convolution needs a CÃ—HÃ—W input` };
      const s = L.s ?? 1, p = L.p ?? 0;
      const h: number = Math.floor((shape.h + 2 * p - L.k) / s) + 1;
      const w: number = Math.floor((shape.w + 2 * p - L.k) / s) + 1;
      if (h <= 0 || w <= 0) return { rows, total, error: `${name}: output size â‰¤ 0 â€” kernel larger than padded input` };
      const params = L.out * shape.c * L.k * L.k + L.out;
      rows.push({ name, op: `Conv ${L.k}Ã—${L.k}, ${L.out} filters, stride ${s}, pad ${p}`, shape: `${L.out}Ã—${h}Ã—${w}`, params, formula: `${L.out}Â·${shape.c}Â·${L.k}Â·${L.k} + ${L.out}` });
      shape = { c: L.out, h, w };
      total += params;
    } else if (L.type === "pool") {
      if (!("c" in shape)) return { rows, total, error: `${name}: pooling needs a CÃ—HÃ—W input` };
      const s = L.s ?? L.k;
      const h: number = Math.floor((shape.h - L.k) / s) + 1, w: number = Math.floor((shape.w - L.k) / s) + 1;
      shape = { c: shape.c, h, w };
      rows.push({ name, op: `${L.mode === "avg" ? "Avg" : "Max"} pool ${L.k}Ã—${L.k}, stride ${s}`, shape: fmtShape(shape), params: 0, formula: "no parameters" });
    } else if (L.type === "adaptivepool") {
      if (!("c" in shape)) return { rows, total, error: `${name}: needs a CÃ—HÃ—W input` };
      shape = { c: shape.c, h: L.size, w: L.size };
      rows.push({ name, op: `Adaptive avg pool â†’ ${L.size}Ã—${L.size}`, shape: fmtShape(shape), params: 0, formula: "no parameters" });
    } else if (L.type === "flatten") {
      const f: number = "c" in shape ? shape.c * shape.h * shape.w : shape.f;
      shape = { f };
      rows.push({ name, op: "Flatten", shape: `${f}`, params: 0, formula: "reshape only" });
    } else if (L.type === "dense") {
      if ("c" in shape) return { rows, total, error: `${name}: dense layer needs a flattened input â€” add Flatten` };
      const params = shape.f * L.out + L.out;
      rows.push({ name, op: `Dense ${shape.f} â†’ ${L.out}`, shape: `${L.out}`, params, formula: `${shape.f}Â·${L.out} + ${L.out}` });
      shape = { f: L.out };
      total += params;
    } else {
      rows.push({ name, op: L.type === "relu" ? "ReLU" : "Dropout", shape: fmtShape(shape), params: 0, formula: "no parameters" });
    }
  }
  return { rows, total };
}

function fmtShape(s: Shape) {
  return "c" in s ? `${s.c}Ã—${s.h}Ã—${s.w}` : `${s.f}`;
}

const conv = (out: number, k: number, p = 0, s = 1, name?: string): LayerSpec => ({ type: "conv", out, k, p, s, name });
const pool = (k = 2, s?: number, mode: "max" | "avg" = "max", name?: string): LayerSpec => ({ type: "pool", k, s, mode, name });
const dense = (out: number, name?: string): LayerSpec => ({ type: "dense", out, name });

const vggBlock = (n: number, ch: number, b: number): LayerSpec[] => [...Array.from({ length: n }, (_, i) => conv(ch, 3, 1, 1, `conv${b}_${i + 1}`)), pool(2, 2, "max", `pool${b}`)];

export const CNN_PRESETS: { id: string; name: string; year: number; input: { c: number; h: number; w: number }; layers: LayerSpec[]; note: string }[] = [
  {
    id: "lenet5",
    name: "LeNet-5",
    year: 1998,
    input: { c: 1, h: 32, w: 32 },
    layers: [conv(6, 5, 0, 1, "C1"), pool(2, 2, "avg", "S2"), conv(16, 5, 0, 1, "C3"), pool(2, 2, "avg", "S4"), { type: "flatten" }, dense(120, "C5/F5"), dense(84, "F6"), dense(10, "output")],
    note: "Modern simplified LeNet-5 (full C3 connectivity, average pooling without trainable coefficients). The 1998 original used a partial C3 connection table and trainable subsampling, giving slightly different counts.",
  },
  {
    id: "alexnet",
    name: "AlexNet (torchvision variant)",
    year: 2012,
    input: { c: 3, h: 224, w: 224 },
    layers: [conv(64, 11, 2, 4, "conv1"), pool(3, 2), conv(192, 5, 2, 1, "conv2"), pool(3, 2), conv(384, 3, 1, 1, "conv3"), conv(256, 3, 1, 1, "conv4"), conv(256, 3, 1, 1, "conv5"), pool(3, 2), { type: "adaptivepool", size: 6 }, { type: "flatten" }, dense(4096, "fc6"), dense(4096, "fc7"), dense(1000, "fc8")],
    note: "Single-GPU variant used by torchvision (64/192 channels in the first layers rather than the paper's split 96/256). Notice ~94% of parameters sit in the fully-connected layers.",
  },
  {
    id: "vgg16",
    name: "VGG-16",
    year: 2014,
    input: { c: 3, h: 224, w: 224 },
    layers: [...vggBlock(2, 64, 1), ...vggBlock(2, 128, 2), ...vggBlock(3, 256, 3), ...vggBlock(3, 512, 4), ...vggBlock(3, 512, 5), { type: "flatten" }, dense(4096, "fc6"), dense(4096, "fc7"), dense(1000, "fc8")],
    note: "Thirteen 3Ã—3 convolutions + three dense layers. Two stacked 3Ã—3 convs have the receptive field of one 5Ã—5 with fewer parameters (2Â·9CÂ² vs 25CÂ²).",
  },
];

/** Parameter count of a BERT/GPT-style Transformer encoder stack (with embeddings). */
export function transformerParams(o: { vocab: number; d: number; layers: number; dff: number; maxPos: number; typeVocab: number; tiedHead: boolean; pooler: boolean }) {
  const { vocab, d, layers, dff, maxPos, typeVocab } = o;
  const embeddings = vocab * d + maxPos * d + typeVocab * d + 2 * d; // + embedding LayerNorm
  const attn = 4 * (d * d + d); // Q, K, V, O projections with bias
  const ffn = d * dff + dff + dff * d + d;
  const norms = 2 * 2 * d;
  const perLayer = attn + ffn + norms;
  const pooler = o.pooler ? d * d + d : 0;
  const head = o.tiedHead ? 0 : vocab * d;
  return { embeddings, attn, ffn, norms, perLayer, encoder: perLayer * layers, pooler, head, total: embeddings + perLayer * layers + pooler + head };
}
