/**
 * Adversarial examples on real 8×8-digit classifiers (pure math, no React).
 *
 * Models (trained by web/scripts/train-adversarial.py, exported to public/models/adversarial.json):
 *   linear  — softmax regression 64 → 10
 *   mlp     — 64 → 32 (ReLU) → 10
 *   mlp_adv — the same MLP trained adversarially with PGD (ε = 0.1)
 *
 * Everything is float64 and mirrors the NumPy reference in the training script exactly:
 *   loss      L(x, y) = −log softmax(f(x))_y
 *   gradient  ∇ₓL by backpropagation:  dL/dlogits = p − onehot(y), then through each layer (Wᵀ, ReLU mask)
 *   FGSM      x' = clip₍₀,₁₎(x + ε·sign(∇ₓL))
 *   PGD       x_{t+1} = clip₍₀,₁₎(clip₍x−ε, x+ε₎(x_t + α·sign(∇ₓL(x_t)))),  x₀ = x  (no random start)
 * `verifyAdversarial` checks the engine against the NumPy verification vectors in the JSON.
 */

export type ModelName = "linear" | "mlp" | "mlp_adv";

export interface LayerJSON {
  W: number[][]; // in × out ("x @ W" orientation)
  b: number[];
}

export interface ModelJSON {
  layers: LayerJSON[];
  n_params: number;
  test_acc: number;
  train_acc: number;
  curves: { fgsm: number[]; pgd: number[] };
}

export interface VerifyJSON {
  model: ModelName;
  index: number;
  label: number;
  loss: number;
  probs: number[];
  grad: number[];
  fgsm_eps: number;
  fgsm: number[];
  fgsm_probs: number[];
  pgd_eps: number;
  pgd_alpha: number;
  pgd_steps: number;
  pgd: number[];
  pgd_probs: number[];
}

export interface AdversarialJSON {
  format: string;
  description: string;
  data: { name: string; n_train: number; n_test: number; split: string };
  adv_training: { eps: number; alpha: number; steps: number; random_start: boolean };
  curve_attack: { eps: number[]; pgd_steps: number; pgd_alpha: string; random_start: boolean };
  models: Record<ModelName, ModelJSON>;
  test: { images: number[][]; labels: number[] };
  verify: VerifyJSON[];
}

export type Classifier = LayerJSON[];

export interface Forward {
  logits: number[];
  probs: number[];
  /** pre-activations of every layer except the last (needed for the ReLU mask) */
  pre: number[][];
}

export function testImage(json: AdversarialJSON, i: number): number[] {
  return json.test.images[i].map((v) => v / 16);
}

export function softmax(z: number[]): number[] {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function forward(model: Classifier, x: number[]): Forward {
  let h = x;
  const pre: number[][] = [];
  model.forEach((layer, li) => {
    const out = layer.b.slice();
    for (let i = 0; i < h.length; i++) {
      const hi = h[i];
      if (hi === 0) continue;
      const row = layer.W[i];
      for (let j = 0; j < out.length; j++) out[j] += hi * row[j];
    }
    if (li < model.length - 1) {
      pre.push(out);
      h = out.map((v) => Math.max(v, 0));
    } else h = out;
  });
  return { logits: h, probs: softmax(h), pre };
}

export function predict(model: Classifier, x: number[]): number {
  const { logits } = forward(model, x);
  return logits.indexOf(Math.max(...logits));
}

/** Cross-entropy loss and its gradient with respect to the input pixels. */
export function lossGrad(model: Classifier, x: number[], y: number): { loss: number; grad: number[]; probs: number[] } {
  const f = forward(model, x);
  let g = f.probs.slice();
  g[y] -= 1; // dL/dlogits = p − onehot(y)
  for (let li = model.length - 1; li >= 0; li--) {
    const W = model[li].W;
    const gin = new Array<number>(W.length).fill(0);
    for (let i = 0; i < W.length; i++) {
      let s = 0;
      for (let j = 0; j < g.length; j++) s += W[i][j] * g[j];
      gin[i] = s;
    }
    g = li > 0 ? gin.map((v, i) => (f.pre[li - 1][i] > 0 ? v : 0)) : gin;
  }
  return { loss: -Math.log(f.probs[y]), grad: g, probs: f.probs };
}

const clip01 = (v: number) => Math.min(Math.max(v, 0), 1);

export function fgsm(model: Classifier, x: number[], y: number, eps: number): number[] {
  const { grad } = lossGrad(model, x, y);
  return x.map((v, i) => clip01(v + eps * Math.sign(grad[i])));
}

export interface PgdStep {
  x: number[];
  loss: number;
  probs: number[];
  grad: number[];
}

/**
 * PGD trajectory: element t is x_t (t = 0 is the clean input) with its loss, probabilities and the
 * gradient at x_t (the direction the next step will take).
 */
export function pgdTrajectory(model: Classifier, x0: number[], y: number, eps: number, alpha: number, steps: number): PgdStep[] {
  const out: PgdStep[] = [];
  let x = x0.slice();
  for (let t = 0; t <= steps; t++) {
    const lg = lossGrad(model, x, y);
    out.push({ x, loss: lg.loss, probs: lg.probs, grad: lg.grad });
    if (t === steps) break;
    x = x.map((v, i) => {
      const stepped = v + alpha * Math.sign(lg.grad[i]);
      return clip01(Math.min(Math.max(stepped, x0[i] - eps), x0[i] + eps));
    });
  }
  return out;
}

export function pgd(model: Classifier, x0: number[], y: number, eps: number, alpha: number, steps: number): number[] {
  const tr = pgdTrajectory(model, x0, y, eps, alpha, steps);
  return tr[tr.length - 1].x;
}

/** Fraction of images still classified correctly after the attack. */
export function robustAccuracy(
  model: Classifier,
  images: number[][],
  labels: number[],
  eps: number,
  attack: "fgsm" | "pgd",
  pgdSteps = 20,
  relAlpha = 2.5,
): number {
  let ok = 0;
  images.forEach((x, i) => {
    const y = labels[i];
    const xa = eps === 0 ? x : attack === "fgsm" ? fgsm(model, x, y, eps) : pgd(model, x, y, eps, (relAlpha * eps) / pgdSteps, pgdSteps);
    if (predict(model, xa) === y) ok++;
  });
  return ok / images.length;
}

export function linfDist(a: number[], b: number[]): number {
  return a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
}

/** Central finite-difference gradient (for testing the analytic one). */
export function numericGrad(model: Classifier, x: number[], y: number, h = 1e-6): number[] {
  const L = (v: number[]) => -Math.log(forward(model, v).probs[y]);
  return x.map((_, i) => {
    const a = x.slice();
    const b = x.slice();
    a[i] += h;
    b[i] -= h;
    return (L(a) - L(b)) / (2 * h);
  });
}

/** Compare the engine with the NumPy verification vectors stored in the JSON. */
export function verifyAdversarial(json: AdversarialJSON): { maxAbsDiff: number; checked: number } {
  let maxAbsDiff = 0;
  let checked = 0;
  const upd = (a: number[], b: number[]) => {
    a.forEach((v, i) => {
      maxAbsDiff = Math.max(maxAbsDiff, Math.abs(v - b[i]));
      checked++;
    });
  };
  for (const v of json.verify) {
    const m = json.models[v.model].layers;
    const x = testImage(json, v.index);
    const lg = lossGrad(m, x, v.label);
    upd([lg.loss], [v.loss]);
    upd(lg.probs, v.probs);
    upd(lg.grad, v.grad);
    const xf = fgsm(m, x, v.label, v.fgsm_eps);
    upd(xf, v.fgsm);
    upd(forward(m, xf).probs, v.fgsm_probs);
    const xp = pgd(m, x, v.label, v.pgd_eps, v.pgd_alpha, v.pgd_steps);
    upd(xp, v.pgd);
    upd(forward(m, xp).probs, v.pgd_probs);
  }
  return { maxAbsDiff, checked };
}
