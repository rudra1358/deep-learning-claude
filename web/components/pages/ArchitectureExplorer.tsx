"use client";

import { useMemo, useState } from "react";
import { Plus, Trash2 } from "lucide-react";
import { CNN_PRESETS, computeCnn, transformerParams, type LayerSpec } from "@/lib/arch";
import { Slider } from "@/components/sims/SimShell";
import { Inline } from "@/components/content/Inline";

const n = (v: number) => v.toLocaleString("en-US");

export function ArchitectureExplorer() {
  const [presetId, setPresetId] = useState("lenet5");
  const [custom, setCustom] = useState<{ input: { c: number; h: number; w: number }; layers: LayerSpec[] } | null>(null);
  const preset = CNN_PRESETS.find((p) => p.id === presetId)!;
  const spec = custom ?? { input: preset.input, layers: preset.layers };
  const result = useMemo(() => computeCnn(spec.input, spec.layers), [spec]);
  const maxParams = Math.max(1, ...result.rows.map((r) => r.params));

  const edit = (fn: (l: LayerSpec[]) => LayerSpec[]) => setCustom({ input: spec.input, layers: fn(spec.layers.slice()) });

  return (
    <div className="space-y-12 py-10">
      <section>
        <div className="mb-4 flex flex-wrap items-center gap-2">
          {CNN_PRESETS.map((p) => (
            <button
              key={p.id}
              onClick={() => {
                setPresetId(p.id);
                setCustom(null);
              }}
              className={`sim-btn ${presetId === p.id && !custom ? "!border-accent !bg-accent-soft" : ""}`}
            >
              {p.name} <span className="text-faint">{p.year}</span>
            </button>
          ))}
          {custom && <span className="text-sm text-gold">edited — this is now your own network</span>}
        </div>
        {!custom && <p className="mb-4 max-w-3xl text-sm text-muted">{preset.note}</p>}

        <div className="mb-3 flex flex-wrap items-center gap-3 text-sm">
          <span className="text-muted">Input C×H×W</span>
          {(["c", "h", "w"] as const).map((k) => (
            <input key={k} type="number" min={1} className="field w-20 font-mono" value={spec.input[k]} onChange={(e) => setCustom({ input: { ...spec.input, [k]: Math.max(1, +e.target.value) }, layers: spec.layers })} aria-label={`Input ${k}`} />
          ))}
          <span className="ml-auto text-muted">
            Total parameters: <strong className="font-mono text-text">{n(result.total)}</strong>
          </span>
        </div>

        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-sm">
            <thead className="bg-surface-2 text-left">
              <tr>
                <th className="px-3 py-2">Layer</th>
                <th className="px-3 py-2">Operation</th>
                <th className="px-3 py-2">Output shape</th>
                <th className="px-3 py-2 text-right">Parameters</th>
                <th className="px-3 py-2">Calculation</th>
                <th className="px-3 py-2" />
              </tr>
            </thead>
            <tbody>
              {result.rows.map((r, i) => (
                <tr key={i} className="border-t border-border">
                  <td className="px-3 py-1.5 font-mono text-xs">{r.name}</td>
                  <td className="px-3 py-1.5">{r.op}</td>
                  <td className="px-3 py-1.5 font-mono text-xs">{r.shape}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-xs">
                    <div className="flex items-center justify-end gap-2">
                      {r.params > 0 && <span className="inline-block h-1.5 rounded bg-accent" style={{ width: `${Math.max(2, (r.params / maxParams) * 80)}px` }} />}
                      {r.params ? n(r.params) : "—"}
                    </div>
                  </td>
                  <td className="px-3 py-1.5 font-mono text-xs text-faint">{r.formula}</td>
                  <td className="px-2">
                    {i > 0 && (
                      <button onClick={() => edit((l) => l.filter((_, k) => k !== i - 1))} className="text-faint hover:text-clay" aria-label={`Remove ${r.name}`}>
                        <Trash2 size={14} />
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {result.error && <div className="mt-2 rounded bg-clay-soft px-3 py-2 text-sm">{result.error}</div>}
        <AddLayer onAdd={(l) => edit((ls) => [...ls, l])} />
      </section>

      <TransformerCalc />
    </div>
  );
}

function AddLayer({ onAdd }: { onAdd: (l: LayerSpec) => void }) {
  const [type, setType] = useState<"conv" | "pool" | "flatten" | "dense">("conv");
  const [out, setOut] = useState(32);
  const [k, setK] = useState(3);
  const [s, setS] = useState(1);
  const [p, setP] = useState(1);
  return (
    <div className="mt-3 flex flex-wrap items-end gap-2 text-sm">
      <select className="field" value={type} onChange={(e) => setType(e.target.value as typeof type)} aria-label="Layer type">
        <option value="conv">Conv2d</option>
        <option value="pool">MaxPool</option>
        <option value="flatten">Flatten</option>
        <option value="dense">Dense</option>
      </select>
      {(type === "conv" || type === "dense") && (
        <label className="flex items-center gap-1 text-muted">
          {type === "conv" ? "filters" : "units"} <input type="number" className="field w-20" value={out} min={1} onChange={(e) => setOut(+e.target.value)} />
        </label>
      )}
      {(type === "conv" || type === "pool") && (
        <>
          <label className="flex items-center gap-1 text-muted">
            k <input type="number" className="field w-14" value={k} min={1} onChange={(e) => setK(+e.target.value)} />
          </label>
          <label className="flex items-center gap-1 text-muted">
            stride <input type="number" className="field w-14" value={s} min={1} onChange={(e) => setS(+e.target.value)} />
          </label>
        </>
      )}
      {type === "conv" && (
        <label className="flex items-center gap-1 text-muted">
          pad <input type="number" className="field w-14" value={p} min={0} onChange={(e) => setP(+e.target.value)} />
        </label>
      )}
      <button
        className="sim-btn"
        onClick={() =>
          onAdd(type === "conv" ? { type, out, k, s, p } : type === "pool" ? { type, k, s } : type === "dense" ? { type, out } : { type })
        }
      >
        <Plus size={14} /> Add layer
      </button>
    </div>
  );
}

const TF_PRESETS = {
  "BERT-base": { vocab: 30522, d: 768, layers: 12, dff: 3072, maxPos: 512, typeVocab: 2, tiedHead: true, pooler: true },
  "BERT-large": { vocab: 30522, d: 1024, layers: 24, dff: 4096, maxPos: 512, typeVocab: 2, tiedHead: true, pooler: true },
  "Tiny (custom)": { vocab: 8000, d: 256, layers: 4, dff: 1024, maxPos: 256, typeVocab: 0, tiedHead: true, pooler: false },
};

function TransformerCalc() {
  const [cfg, setCfg] = useState(TF_PRESETS["BERT-base"]);
  const r = transformerParams(cfg);
  const set = (k: keyof typeof cfg, v: number) => setCfg({ ...cfg, [k]: v });
  const parts = [
    ["Embeddings (token + position + type + LayerNorm)", r.embeddings],
    [`Encoder layers (${cfg.layers} × ${n(r.perLayer)})`, r.encoder],
    ["Pooler", r.pooler],
    ["Untied LM head", r.head],
  ] as const;
  return (
    <section>
      <h2 className="font-serif text-2xl font-semibold">Transformer encoder: count the parameters</h2>
      <p className="mt-1 max-w-3xl text-sm text-muted">
        <Inline text={String.raw`Per layer: attention projections $4(d^2+d)$, feed-forward $2\,d\,d_{ff} + d_{ff} + d$, two LayerNorms $4d$. With BERT-base settings you get 109,482,240 — exactly the size of the released BERT-base model.`} />
      </p>
      <div className="mt-4 flex flex-wrap gap-2">
        {Object.entries(TF_PRESETS).map(([name, c]) => (
          <button key={name} onClick={() => setCfg(c)} className="sim-btn">
            {name}
          </button>
        ))}
      </div>
      <div className="mt-4 grid gap-6 md:grid-cols-2">
        <div className="space-y-2">
          <Slider label="d_model" min={64} max={2048} step={64} value={cfg.d} onChange={(v) => set("d", v)} fmt={(v) => String(v)} />
          <Slider label="layers" min={1} max={48} step={1} value={cfg.layers} onChange={(v) => set("layers", v)} fmt={(v) => String(v)} />
          <Slider label="d_ff" min={128} max={8192} step={128} value={cfg.dff} onChange={(v) => set("dff", v)} fmt={(v) => String(v)} />
          <Slider label="vocab" min={1000} max={100000} step={1000} value={cfg.vocab} onChange={(v) => set("vocab", v)} fmt={(v) => String(v)} />
          <Slider label="max positions" min={64} max={8192} step={64} value={cfg.maxPos} onChange={(v) => set("maxPos", v)} fmt={(v) => String(v)} />
        </div>
        <div className="card p-4 text-sm">
          {parts.map(([label, v]) => (
            <div key={label} className="flex justify-between border-b border-border py-1.5 last:border-0">
              <span className="text-muted">{label}</span>
              <span className="font-mono">{n(v)}</span>
            </div>
          ))}
          <div className="mt-2 flex justify-between font-medium">
            <span>Total</span>
            <span className="font-mono">{n(r.total)}</span>
          </div>
          <div className="mt-1 text-xs text-faint">≈ {(r.total / 1e6).toFixed(1)} M parameters · {((r.total * 4) / 2 ** 20).toFixed(0)} MiB in fp32</div>
        </div>
      </div>
    </section>
  );
}
