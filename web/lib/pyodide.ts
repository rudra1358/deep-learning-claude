/**
 * Browser-side Python execution via Pyodide (CPython compiled to WebAssembly).
 *
 * Safety: code runs entirely inside the learner's own browser sandbox —
 * nothing is executed on our servers. Limitations: NumPy, Matplotlib,
 * scikit-learn and pandas are available; PyTorch/TensorFlow are NOT
 * (no WebAssembly builds), so framework examples are shown with expected
 * output and should be run locally.
 */

const PYODIDE_VERSION = "0.27.7";
const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

type PyodideLike = {
  loadPackagesFromImports: (code: string) => Promise<void>;
  runPythonAsync: (code: string) => Promise<unknown>;
  setStdout: (o: { batched: (s: string) => void }) => void;
  setStderr: (o: { batched: (s: string) => void }) => void;
};

declare global {
  interface Window {
    loadPyodide?: (opts: { indexURL: string }) => Promise<PyodideLike>;
  }
}

let pyodidePromise: Promise<PyodideLike> | null = null;

function loadScript(src: string) {
  return new Promise<void>((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) return resolve();
    const s = document.createElement("script");
    s.src = src;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("Failed to load Pyodide from CDN (are you offline?)"));
    document.head.appendChild(s);
  });
}

export function getPyodide() {
  if (!pyodidePromise) {
    pyodidePromise = (async () => {
      await loadScript(`${INDEX_URL}pyodide.js`);
      if (!window.loadPyodide) throw new Error("Pyodide failed to initialise");
      return window.loadPyodide({ indexURL: INDEX_URL });
    })();
    pyodidePromise.catch(() => (pyodidePromise = null));
  }
  return pyodidePromise;
}

export interface RunResult {
  stdout: string;
  error?: string;
  ms: number;
  images?: string[]; // base64 PNGs of any Matplotlib figures created
}

const CAPTURE_FIGURES = `
import sys, json
_imgs = []
if 'matplotlib.pyplot' in sys.modules:
    import io, base64
    import matplotlib.pyplot as _plt
    for _n in _plt.get_fignums():
        _buf = io.BytesIO()
        _plt.figure(_n).savefig(_buf, format='png', dpi=110, bbox_inches='tight')
        _imgs.append(base64.b64encode(_buf.getvalue()).decode())
    _plt.close('all')
json.dumps(_imgs)
`;

const UNSUPPORTED = /^\s*(import|from)\s+(torch|tensorflow|keras|transformers|jax)\b/m;

export async function runPython(code: string): Promise<RunResult> {
  const t0 = performance.now();
  if (UNSUPPORTED.test(code)) {
    return {
      stdout: "",
      error:
        "This example imports PyTorch/TensorFlow, which cannot run in the browser (no WebAssembly build exists).\nRun it locally: python -m venv .venv && pip install torch numpy  — then python example.py",
      ms: 0,
    };
  }
  let out = "";
  try {
    const py = await getPyodide();
    py.setStdout({ batched: () => {} });
    await py.loadPackagesFromImports(code); // its "Loading numpy…" chatter is discarded
    py.setStdout({ batched: (s) => (out += s + "\n") });
    py.setStderr({ batched: (s) => (out += s + "\n") });
    // Use a non-interactive matplotlib backend so plotting code doesn't crash.
    await py.runPythonAsync("import os; os.environ['MPLBACKEND']='Agg'");
    if (/matplotlib/.test(code)) {
      // Figures are captured after the run, so plt.show() becomes a no-op instead of warning.
      await py.runPythonAsync("import matplotlib.pyplot as _p; _p.show = lambda *a, **k: None");
    }
    await py.runPythonAsync(code);
    const imgs = JSON.parse(String(await py.runPythonAsync(CAPTURE_FIGURES))) as string[];
    return { stdout: out.trimEnd(), ms: performance.now() - t0, images: imgs };
  } catch (e) {
    return { stdout: out.trimEnd(), error: String(e), ms: performance.now() - t0 };
  }
}
