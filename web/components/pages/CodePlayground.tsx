"use client";

import { useRef, useState } from "react";
import { Play, RotateCcw, Info } from "lucide-react";
import { runPython, type RunResult } from "@/lib/pyodide";

const EXAMPLES: Record<string, string> = {
  "Neuron forward pass": `import numpy as np

# One artificial neuron: weighted sum of inputs + bias, then sigmoid.
x = np.array([0.5, -1.0, 2.0])      # inputs
w = np.array([0.4, 0.3, -0.2])      # weights
b = 0.1                              # bias

z = w @ x + b                        # weighted sum (dot product)
a = 1 / (1 + np.exp(-z))             # sigmoid activation
print(f"z = {z:.4f}")
print(f"a = sigmoid(z) = {a:.4f}")
`,
  "Train a 2-layer network on XOR": `import numpy as np

rng = np.random.default_rng(0)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

# With only 4 hidden units, an unlucky initialisation (e.g. seed 0) can get stuck
# predicting 0.5 for two inputs. 8 hidden units converged for all 20 seeds we tried.
HIDDEN, LR, EPOCHS = 8, 1.0, 3000
W1 = rng.normal(0, 1, (2, HIDDEN)); b1 = np.zeros(HIDDEN)
W2 = rng.normal(0, 1, (HIDDEN, 1)); b2 = np.zeros(1)
sigmoid = lambda z: 1 / (1 + np.exp(-z))

for epoch in range(EPOCHS):
    # forward
    h = np.tanh(X @ W1 + b1)
    p = sigmoid(h @ W2 + b2)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    # backward (BCE + sigmoid => dL/dz2 = (p - y) / N)
    dz2 = (p - y) / len(X)
    dW2 = h.T @ dz2; db2 = dz2.sum(0)
    dz1 = (dz2 @ W2.T) * (1 - h ** 2)
    dW1 = X.T @ dz1; db1 = dz1.sum(0)
    # update
    W1 -= LR * dW1; b1 -= LR * db1; W2 -= LR * dW2; b2 -= LR * db2
    if epoch % 500 == 0:
        print(f"epoch {epoch:4d}  loss {loss:.4f}")

print("predictions:", p.ravel().round(3))
`,
  "Plot activation functions": `import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 400)
fns = {
    "sigmoid": 1 / (1 + np.exp(-x)),
    "tanh": np.tanh(x),
    "ReLU": np.maximum(0, x),
    "GELU (tanh approx.)": 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))),
}
fig, ax = plt.subplots(figsize=(6, 3.6))
for name, y in fns.items():
    ax.plot(x, y, label=name)
ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
ax.legend(); ax.set_title("Activation functions")
plt.show()
`,
  "2-D convolution from scratch": `import numpy as np

image = np.array([[0, 0, 0, 9, 9, 9],
                  [0, 0, 0, 9, 9, 9],
                  [0, 0, 0, 9, 9, 9],
                  [0, 0, 0, 9, 9, 9]], dtype=float)
kernel = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=float)   # Sobel: responds to vertical edges

def conv2d(x, k, stride=1, pad=0):
    x = np.pad(x, pad)
    kh, kw = k.shape
    oh = (x.shape[0] - kh) // stride + 1
    ow = (x.shape[1] - kw) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            window = x[i*stride:i*stride+kh, j*stride:j*stride+kw]
            out[i, j] = np.sum(window * k)       # multiply-accumulate
    return out

print(conv2d(image, kernel))
`,
  "Softmax & cross-entropy": `import numpy as np

logits = np.array([2.0, 1.0, 0.1])
target = 0  # index of the true class

shifted = logits - logits.max()           # numerical stability
probs = np.exp(shifted) / np.exp(shifted).sum()
loss = -np.log(probs[target])

print("probabilities:", probs.round(4), "sum =", probs.sum())
print("cross-entropy loss:", round(loss, 4))
grad = probs.copy(); grad[target] -= 1    # dL/dlogits = p - one_hot(y)
print("gradient wrt logits:", grad.round(4))
`,
};

export function CodePlayground() {
  const [name, setName] = useState(Object.keys(EXAMPLES)[0]);
  const [code, setCode] = useState(EXAMPLES[name]);
  const [res, setRes] = useState<RunResult | null>(null);
  const [running, setRunning] = useState(false);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const lines = code.split("\n").length;

  async function run() {
    setRunning(true);
    setRes({ stdout: "Loading the Python runtime (first run downloads ~10 MB)…", ms: 0 });
    setRes(await runPython(code));
    setRunning(false);
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      void run();
    } else if (e.key === "Tab") {
      e.preventDefault();
      const ta = e.currentTarget;
      const { selectionStart: s, selectionEnd: en } = ta;
      const next = code.slice(0, s) + "    " + code.slice(en);
      setCode(next);
      requestAnimationFrame(() => {
        ta.selectionStart = ta.selectionEnd = s + 4;
      });
    }
  }

  return (
    <div className="py-8">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <select
          className="field"
          value={name}
          onChange={(e) => {
            setName(e.target.value);
            setCode(EXAMPLES[e.target.value]);
            setRes(null);
          }}
          aria-label="Example"
        >
          {Object.keys(EXAMPLES).map((k) => (
            <option key={k}>{k}</option>
          ))}
        </select>
        <button className="sim-btn primary" onClick={run} disabled={running}>
          <Play size={14} /> {running ? "Running…" : "Run"} <kbd className="ml-1 text-[0.65rem] opacity-70">Ctrl ↵</kbd>
        </button>
        <button className="sim-btn" onClick={() => setCode(EXAMPLES[name])}>
          <RotateCcw size={14} /> Reset code
        </button>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="flex overflow-hidden rounded-lg border border-border bg-code font-mono text-[0.82rem] leading-6">
          <div className="select-none border-r border-border px-2 py-3 text-right text-faint" aria-hidden>
            {Array.from({ length: lines }, (_, i) => (
              <div key={i}>{i + 1}</div>
            ))}
          </div>
          <textarea ref={taRef} value={code} onChange={(e) => setCode(e.target.value)} onKeyDown={onKeyDown} spellCheck={false} className="min-h-[460px] flex-1 resize-none bg-transparent px-3 py-3 outline-none" style={{ height: `${Math.max(460, lines * 24 + 24)}px` }} aria-label="Python code editor" />
        </div>
        <div className="flex flex-col rounded-lg border border-border bg-surface">
          <div className="flex justify-between border-b border-border px-3 py-1.5 text-xs text-faint">
            <span>Output</span>
            {res && res.ms > 0 && <span>{(res.ms / 1000).toFixed(2)} s</span>}
          </div>
          <div className="flex-1 overflow-auto p-3">
            {!res && <div className="text-sm text-faint">Press Run to execute.</div>}
            {res && <pre className="font-mono text-[0.8rem] whitespace-pre-wrap">{res.stdout}</pre>}
            {res?.error && <pre className="mt-2 rounded bg-clay-soft p-2 font-mono text-[0.78rem] whitespace-pre-wrap">{res.error}</pre>}
            {res?.images?.map((src, i) => (
              // eslint-disable-next-line @next/next/no-img-element
              <img key={i} src={`data:image/png;base64,${src}`} alt={`Figure ${i + 1}`} className="mt-3 max-w-full rounded bg-white" />
            ))}
          </div>
        </div>
      </div>
      <div className="mt-4 flex gap-2 rounded-lg border border-border bg-surface-2 p-3 text-sm text-muted">
        <Info size={16} className="mt-0.5 shrink-0" />
        <div>
          <strong className="text-text">Execution limits.</strong> Runs on your CPU, single-threaded, inside the browser sandbox. NumPy, Matplotlib, pandas and scikit-learn are available. PyTorch and TensorFlow have no WebAssembly build, so the PyTorch/Keras examples in lessons must be run locally (each lesson shows their real output). For GPU work use a local machine or a cloud notebook.
        </div>
      </div>
    </div>
  );
}
