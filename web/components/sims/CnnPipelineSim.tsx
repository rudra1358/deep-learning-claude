"use client";

import { useEffect, useMemo, useRef, useState, type ChangeEvent, type ReactNode } from "react";
import { Controls, Narration, SimShell, useStepper } from "./SimShell";
import { convParams, convPlanes, maxPoolPlane, outputSize, planeStats, reluPlane, type Plane } from "@/lib/sim/conv";

const SIZE = 64;

/* ------------------------------------------------------------------ */
/* Hand-designed kernel banks (NOT learned weights)                    */
/* ------------------------------------------------------------------ */

const L1: { name: string; k: number[] }[] = [
  { name: "Vertical edge (Sobel-x)", k: [-1, 0, 1, -2, 0, 2, -1, 0, 1] },
  { name: "Horizontal edge (Sobel-y)", k: [-1, -2, -1, 0, 0, 0, 1, 2, 1] },
  { name: "Diagonal edge ↘", k: [-2, -1, 0, -1, 0, 1, 0, 1, 2] },
  { name: "Diagonal edge ↙", k: [0, -1, -2, 1, 0, -1, 2, 1, 0] },
  { name: "Centre-surround (spots)", k: [-1, -1, -1, -1, 8, -1, -1, -1, -1] },
  { name: "Gaussian blur", k: [1, 2, 1, 2, 4, 2, 1, 2, 1].map((v) => v / 16) },
];

const Z = [0, 0, 0, 0, 0, 0, 0, 0, 0];
const BOX = Z.map(() => 1 / 9);
const VCOL = [0, 1, 0, 0, 1, 0, 0, 1, 0].map((v) => v / 3);
const HROW = [0, 0, 0, 1, 1, 1, 0, 0, 0].map((v) => v / 3);
const sc = (k: number[], a: number) => k.map((v) => v * a);

/** Layer 2: each filter has one 3×3 slice per layer-1 channel (shape 6×3×3). */
const L2: { name: string; recipe: string; w: number[][] }[] = [
  { name: "Long vertical edges", recipe: "vertical-line average of the Sobel-x map − ½·box of the Sobel-y map", w: [VCOL, sc(BOX, -0.5), Z, Z, Z, Z] },
  { name: "Long horizontal edges", recipe: "horizontal-line average of the Sobel-y map − ½·box of the Sobel-x map", w: [sc(BOX, -0.5), HROW, Z, Z, Z, Z] },
  { name: "Edge energy (any orientation)", recipe: "¼·box over each of the four edge maps", w: [sc(BOX, 0.25), sc(BOX, 0.25), sc(BOX, 0.25), sc(BOX, 0.25), Z, Z] },
  { name: "Spots minus edges", recipe: "box of the centre-surround map − ½·box of the Sobel-x and Sobel-y maps", w: [sc(BOX, -0.5), sc(BOX, -0.5), Z, Z, BOX, Z] },
];

/* ------------------------------------------------------------------ */
/* Built-in procedurally generated sample images (64×64, values 0..1)  */
/* ------------------------------------------------------------------ */

function inTri(px: number, py: number, a: number[], b: number[], c: number[]) {
  const s = (p: number[], q: number[], r: number[]) => (p[0] - r[0]) * (q[1] - r[1]) - (q[0] - r[0]) * (p[1] - r[1]);
  const d1 = s([px, py], a, b);
  const d2 = s([px, py], b, c);
  const d3 = s([px, py], c, a);
  const neg = d1 < 0 || d2 < 0 || d3 < 0;
  const pos = d1 > 0 || d2 > 0 || d3 > 0;
  return !(neg && pos);
}

function makeSample(id: SampleId): Plane {
  const d = new Float32Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++)
    for (let x = 0; x < SIZE; x++) {
      let v = 0;
      if (id === "shapes") {
        v = 0.12;
        if ((x - 19) ** 2 + (y - 20) ** 2 <= 11 ** 2) v = 0.9;
        if (x >= 30 && x <= 54 && y >= 36 && y <= 56) v = 0.6;
        if (inTri(x, y, [46, 5], [61, 28], [33, 28])) v = 1;
        if (x >= 6 && x <= 22 && y >= 44 && y <= 47) v = 0.75; // a thin horizontal bar
      } else if (id === "house") {
        v = 0.85 - y * 0.006; // sky gradient
        if ((x - 54) ** 2 + (y - 9) ** 2 <= 25) v = 1; // sun
        if (y >= 52) v = 0.3; // ground
        if (x >= 14 && x <= 50 && y >= 30 && y < 52) v = 0.55; // walls
        if (inTri(x, y, [9, 30], [55, 30], [32, 11])) v = 0.22; // roof
        if (x >= 28 && x <= 36 && y >= 40 && y < 52) v = 0.08; // door
        if (((x >= 18 && x <= 24) || (x >= 40 && x <= 46)) && y >= 34 && y <= 40) v = 0.95; // windows
      } else {
        // concentric rings on the left, diagonal stripes on the right
        if (x < 32) {
          const r = Math.hypot(x - 16, y - 32);
          v = 0.5 + 0.5 * Math.cos(r * 0.75);
        } else {
          v = Math.floor((x + y) / 6) % 2 === 0 ? 0.9 : 0.15;
        }
      }
      d[y * SIZE + x] = v;
    }
  return { data: d, h: SIZE, w: SIZE };
}

type SampleId = "shapes" | "house" | "rings";
const SAMPLES: { id: SampleId; label: string }[] = [
  { id: "shapes", label: "Shapes" },
  { id: "house", label: "House" },
  { id: "rings", label: "Rings & stripes" },
];

/** Stretch to [0, 1] (simple contrast normalisation so every image uses the full range). */
function normalise(p: Plane): Plane {
  const { min, max } = planeStats(p);
  const r = max - min || 1;
  const out = new Float32Array(p.data.length);
  for (let i = 0; i < out.length; i++) out[i] = (p.data[i] - min) / r;
  return { data: out, h: p.h, w: p.w };
}

/** Centre-crop to a square, resize to 64×64 on a canvas, convert to grayscale luminance. */
function imageToPlane(img: HTMLImageElement): Plane {
  const c = document.createElement("canvas");
  c.width = SIZE;
  c.height = SIZE;
  const ctx = c.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D not available");
  const side = Math.min(img.naturalWidth, img.naturalHeight);
  const sx = (img.naturalWidth - side) / 2;
  const sy = (img.naturalHeight - side) / 2;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, sx, sy, side, side, 0, 0, SIZE, SIZE);
  const px = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const d = new Float32Array(SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) d[i] = (0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2]) / 255;
  return { data: d, h: SIZE, w: SIZE };
}

/* ------------------------------------------------------------------ */
/* Theme-aware colours for canvas (canvas cannot use CSS variables)    */
/* ------------------------------------------------------------------ */

type RGB = [number, number, number];
interface Palette {
  accent: RGB;
  clay: RGB;
  surface: RGB;
}

function parseColor(s: string, fallback: RGB): RGB {
  const t = s.trim();
  let m = /^#([0-9a-f]{3})$/i.exec(t);
  if (m) return [0, 1, 2].map((i) => parseInt(m![1][i] + m![1][i], 16)) as RGB;
  m = /^#([0-9a-f]{6})$/i.exec(t);
  if (m) return [0, 2, 4].map((i) => parseInt(m![1].slice(i, i + 2), 16)) as RGB;
  m = /rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)/i.exec(t);
  if (m) return [+m[1], +m[2], +m[3]];
  return fallback;
}

function readPalette(): Palette {
  const cs = getComputedStyle(document.documentElement);
  return {
    accent: parseColor(cs.getPropertyValue("--accent"), [61, 122, 107]),
    clay: parseColor(cs.getPropertyValue("--clay"), [181, 111, 85]),
    surface: parseColor(cs.getPropertyValue("--surface"), [255, 255, 255]),
  };
}

function usePalette(): Palette | null {
  const [pal, setPal] = useState<Palette | null>(null);
  useEffect(() => {
    setPal(readPalette());
    const mo = new MutationObserver(() => setPal(readPalette()));
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class", "style"] });
    return () => mo.disconnect();
  }, []);
  return pal;
}

const mix = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

type MapMode = "gray" | "seq" | "signed";

function MapCanvas({ plane, mode, pal, width, label }: { plane: Plane; mode: MapMode; pal: Palette | null; width: number; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || !pal) return;
    c.width = plane.w;
    c.height = plane.h;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(plane.w, plane.h);
    const { min, max } = planeStats(plane);
    const maxAbs = Math.max(Math.abs(min), Math.abs(max)) || 1;
    for (let i = 0; i < plane.data.length; i++) {
      const v = plane.data[i];
      let rgb: RGB;
      if (mode === "gray") {
        const g = Math.max(0, Math.min(1, v)) * 255;
        rgb = [g, g, g];
      } else if (mode === "seq") {
        rgb = mix(pal.surface, pal.accent, max > 0 ? Math.max(0, v) / max : 0);
      } else {
        rgb = v >= 0 ? mix(pal.surface, pal.accent, v / maxAbs) : mix(pal.surface, pal.clay, -v / maxAbs);
      }
      img.data[4 * i] = rgb[0];
      img.data[4 * i + 1] = rgb[1];
      img.data[4 * i + 2] = rgb[2];
      img.data[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [plane, mode, pal]);
  return <canvas ref={ref} role="img" aria-label={label} className="block rounded-sm border border-border" style={{ width, height: width, imageRendering: "pixelated" }} />;
}

function KernelTile({ k }: { k: number[] }) {
  const m = Math.max(...k.map(Math.abs)) || 1;
  return (
    <div className="grid grid-cols-3 gap-px font-mono text-[9px] leading-none" aria-label="Kernel weights">
      {k.map((v, i) => {
        const r = Math.round(v * 100) / 100;
        const pct = Math.round(15 + (Math.abs(v) / m) * 60);
        return (
          <span
            key={i}
            className="flex h-[17px] items-center justify-center rounded-[2px]"
            style={{ background: v === 0 ? "var(--surface-2)" : `color-mix(in srgb, ${v > 0 ? "var(--accent)" : "var(--clay)"} ${pct}%, transparent)` }}
          >
            {Number.isInteger(r) ? r : r.toFixed(2).replace(/^(-?)0\./, "$1.")}
          </span>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Simulator                                                           */
/* ------------------------------------------------------------------ */

const STAGES = ["Input", "Conv 1", "ReLU 1", "Max-pool 1", "Conv 2", "ReLU 2", "Max-pool 2"] as const;

const shape = (c: number, p: Plane) => `${c} × ${p.h} × ${p.w}`;

export default function CnnPipelineSim() {
  const [sample, setSample] = useState<SampleId | "upload">("shapes");
  const [uploaded, setUploaded] = useState<Plane | null>(null);
  const [uploadName, setUploadName] = useState("");
  const [error, setError] = useState("");
  const [stride, setStride] = useState(1);
  const [padding, setPadding] = useState(0);
  const pal = usePalette();
  const s = useStepper(STAGES.length, { baseMs: 1600 });
  const stage = s.step;

  const raw = useMemo(() => (sample === "upload" && uploaded ? uploaded : makeSample(sample === "upload" ? "shapes" : sample)), [sample, uploaded]);

  const net = useMemo(() => {
    const input = normalise(raw);
    const conv1 = convPlanes([input], L1.map((f) => [f.k]), L1.map(() => 0), 3, stride, padding);
    const relu1 = conv1.map(reluPlane);
    const pool1 = relu1.map((p) => maxPoolPlane(p));
    const conv2 = convPlanes(pool1, L2.map((f) => f.w), L2.map(() => 0), 3, 1, 1);
    const relu2 = conv2.map(reluPlane);
    const pool2 = relu2.map((p) => maxPoolPlane(p));
    return { input, conv1, relu1, pool1, conv2, relu2, pool2 };
  }, [raw, stride, padding]);

  const onFile = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setError("");
    const url = URL.createObjectURL(f);
    const img = new Image();
    img.onload = () => {
      try {
        setUploaded(imageToPlane(img));
        setUploadName(f.name);
        setSample("upload");
        s.reset();
      } catch (err) {
        setError(err instanceof Error ? err.message : "Could not read that image.");
      } finally {
        URL.revokeObjectURL(url);
      }
    };
    img.onerror = () => {
      setError("That file does not look like an image we can open.");
      URL.revokeObjectURL(url);
    };
    img.src = url;
  };

  const zeroFrac = (ps: Plane[]) => ps.reduce((a, p) => a + planeStats(p).zeroFrac, 0) / ps.length;
  const conv1Size = outputSize(SIZE, 3, padding, stride);
  const p1 = net.pool1[0];
  const p2 = net.pool2[0];
  const inputCount = SIZE * SIZE;
  const finalCount = net.pool2.length * p2.h * p2.w;
  // receptive field of one final value through (conv3 stride s) → pool2 → conv3 → pool2:
  // r_l = r_{l-1} + (k_l − 1)·j_{l-1},  j_l = j_{l-1}·s_l
  const rf = (() => {
    let r = 1;
    let j = 1;
    const layers: [number, number][] = [
      [3, stride],
      [2, 2],
      [3, 1],
      [2, 2],
    ];
    for (const [k, st] of layers) {
      r += (k - 1) * j;
      j *= st;
    }
    return r;
  })();

  const narration: { title: string; text: string }[] = [
    {
      title: "Stage 0: the input tensor",
      text: `We crop the image to a square, shrink it to ${SIZE}×${SIZE}, turn it grey (0.299R + 0.587G + 0.114B), and stretch it to [0, 1]. Now it is a tensor (a grid of numbers) with one channel: 1 × ${SIZE} × ${SIZE} = ${inputCount} numbers. Press Step to apply the first layer.`,
    },
    {
      title: "Stage 1: convolution with 6 hand-made kernels",
      text: `Each 3×3 kernel (a small grid of weights) slides over the image with stride ${stride} and padding ${padding}: ⌊(${SIZE} + 2·${padding} − 3)/${stride}⌋ + 1 = ${conv1Size}. So the output is ${shape(6, net.conv1[0])}. Teal = positive response, clay = negative. Edge kernels add up to 0, so flat areas give 0 and only changes in brightness light up. The sign tells you which way the change goes.${padding > 0 ? " With padding 1, the ring of zeros acts like a dark frame around the image. So the edge maps also light up along the border, a side effect of zero padding." : ""} This layer would have ${convParams(1, 6, 3)} parameters (6·1·3·3 weights + 6 biases).`,
    },
    {
      title: "Stage 2: ReLU",
      text: `ReLU turns every negative value into 0. Now ${(zeroFrac(net.relu1) * 100).toFixed(1)}% of the ${6 * net.relu1[0].h * net.relu1[0].w} activations are exactly zero. Notice each edge map keeps only ONE direction of change (e.g. dark→bright going right for Sobel-x). The opposite edge is thrown away. That is why trained CNNs usually learn both directions as separate filters.`,
    },
    {
      title: "Stage 3: 2×2 max-pooling",
      text: `Each map keeps only the biggest value in every 2×2 block (blocks do not overlap): ${net.relu1[0].h} → ⌊(${net.relu1[0].h} − 2)/2⌋ + 1 = ${p1.h}, giving ${shape(6, p1)}. A strong response anywhere in the block survives. So small shifts matter a bit less, and there are 4× fewer values.`,
    },
    {
      title: "Stage 4: convolution over all 6 channels",
      text: `Every layer-2 filter has shape 6×3×3. It reads a 3×3 patch from ALL six pooled maps at once and adds up the results. With padding 1 and stride 1 the size stays the same: ${shape(4, net.conv2[0])}. These filters combine layer-1 features (e.g. "vertical edge here AND not a horizontal edge"). Parameters: ${convParams(6, 4, 3)} (4·6·3·3 + 4).`,
    },
    {
      title: "Stage 5: ReLU",
      text: `Negative combinations are thrown away again: ${(zeroFrac(net.relu2) * 100).toFixed(1)}% of layer-2 activations are zero. Layer 2 now responds to patterns made of patterns. This is how a feature hierarchy starts: simple features build into complex ones.`,
    },
    {
      title: "Stage 6: max-pool gives the final features",
      text: `Output: ${shape(4, p2)} = ${finalCount} numbers, versus ${inputCount} input pixels. Each final value depends on a ${rf}×${rf}-pixel patch of the input (its receptive field). So it describes a region, not one pixel. In a real CNN, this tensor would be flattened and fed to a classifier. And every kernel above would be learned by backpropagation, not written by hand.`,
    },
  ];

  const cur = narration[stage];
  const sel = "field py-1 text-xs";

  const stageBox = (idx: number, children: ReactNode, label: string) =>
    stage >= idx ? (
      <div
        key={idx}
        className="rounded-lg border p-2"
        style={{ borderColor: stage === idx ? "var(--gold)" : "var(--border)", borderWidth: stage === idx ? 2 : 1 }}
      >
        <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-2 text-xs">
          <span className="font-medium">
            {idx}. {STAGES[idx]}
          </span>
          <span className="font-mono text-muted">{label}</span>
        </div>
        {children}
      </div>
    ) : null;

  const mapRow = (planes: Plane[], names: string[], mode: MapMode, width: number, kernels?: number[][]) => (
    <div className="flex flex-wrap gap-2">
      {planes.map((p, i) => (
        <figure key={i} className="flex flex-col gap-1" style={{ width }}>
          <MapCanvas plane={p} mode={mode} pal={pal} width={width} label={`${names[i]} feature map, ${p.h} by ${p.w}`} />
          <figcaption className="text-[10px] leading-tight text-muted">{names[i]}</figcaption>
          {kernels && <KernelTile k={kernels[i]} />}
        </figure>
      ))}
    </div>
  );

  const l1Names = L1.map((f) => f.name);
  const l2Names = L2.map((f) => f.name);

  return (
    <SimShell
      title="CNN Feature Extraction"
      subtitle="A real two-layer CNN runs on your image. The kernels are hand-made (not learned), so you can see what each one picks out."
      controls={<Controls s={s} />}
      aside={
        <div className="flex flex-col gap-3">
          <Narration title={cur.title}>{cur.text}</Narration>
          <div className="rounded-md border border-border p-2 font-mono text-[11px] leading-relaxed">
            <div className="mb-1 font-sans text-xs font-medium text-muted">Shapes (C × H × W)</div>
            {[
              ["Input", shape(1, net.input)],
              ["Conv 1", shape(6, net.conv1[0])],
              ["ReLU 1", shape(6, net.relu1[0])],
              ["Pool 1", shape(6, p1)],
              ["Conv 2", shape(4, net.conv2[0])],
              ["ReLU 2", shape(4, net.relu2[0])],
              ["Pool 2", shape(4, p2)],
            ].map(([n, sh], i) => (
              <div key={n} className={`flex justify-between ${i === stage ? "font-semibold text-accent" : i > stage ? "text-faint" : ""}`}>
                <span>{n}</span>
                <span>{sh}</span>
              </div>
            ))}
          </div>
          <p className="text-xs text-faint">
            Colours: grey for the image. For maps before ReLU, teal = positive and clay = negative. Each map is scaled to its own max, so compare patterns between maps, not brightness.
          </p>
        </div>
      }
    >
      <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
        <label className="flex items-center gap-2">
          <span className="text-muted">Image</span>
          <select
            className={sel}
            value={sample}
            onChange={(e) => {
              setSample(e.target.value as SampleId | "upload");
              s.reset();
            }}
            aria-label="Sample image"
          >
            {SAMPLES.map((x) => (
              <option key={x.id} value={x.id}>
                {x.label}
              </option>
            ))}
            {uploaded && <option value="upload">Uploaded: {uploadName.slice(0, 24)}</option>}
          </select>
        </label>
        <label className="sim-btn cursor-pointer">
          Upload image…
          <input type="file" accept="image/*" className="sr-only" onChange={onFile} aria-label="Upload an image" />
        </label>
        <label className="flex items-center gap-2">
          <span className="text-muted">Layer-1 stride</span>
          <select className={sel} value={stride} onChange={(e) => setStride(+e.target.value)} aria-label="Layer 1 stride">
            <option value={1}>1</option>
            <option value={2}>2</option>
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="text-muted">Layer-1 padding</span>
          <select className={sel} value={padding} onChange={(e) => setPadding(+e.target.value)} aria-label="Layer 1 padding">
            <option value={0}>0 (valid)</option>
            <option value={1}>1 (same at stride 1)</option>
          </select>
        </label>
        <button className="sim-btn" onClick={() => s.setStep(STAGES.length - 1)} aria-label="Show all stages">
          Show all
        </button>
      </div>
      {error && <p className="mb-2 text-xs text-clay">{error}</p>}
      <p className="mb-3 text-[11px] text-faint">Your image never leaves your browser. Everything runs on this page.</p>

      <div className="flex flex-col gap-3">
        {stageBox(0, <MapCanvas plane={net.input} mode="gray" pal={pal} width={128} label="Input image, grayscale 64 by 64" />, shape(1, net.input))}
        {stageBox(1, mapRow(net.conv1, l1Names, "signed", 84, L1.map((f) => f.k)), `${shape(6, net.conv1[0])}  ·  k=3, s=${stride}, p=${padding}`)}
        {stageBox(2, mapRow(net.relu1, l1Names, "seq", 84), shape(6, net.relu1[0]))}
        {stageBox(3, mapRow(net.pool1, l1Names, "seq", 84), `${shape(6, p1)}  ·  2×2, s=2`)}
        {stageBox(
          4,
          <>
            {mapRow(net.conv2, l2Names, "signed", 96)}
            <ul className="mt-2 list-disc pl-4 text-[11px] text-muted">
              {L2.map((f) => (
                <li key={f.name}>
                  <span className="text-text">{f.name}</span>: {f.recipe}
                </li>
              ))}
            </ul>
          </>,
          `${shape(4, net.conv2[0])}  ·  kernels 4×6×3×3, s=1, p=1`,
        )}
        {stageBox(5, mapRow(net.relu2, l2Names, "seq", 96), shape(4, net.relu2[0]))}
        {stageBox(6, mapRow(net.pool2, l2Names, "seq", 96), `${shape(4, p2)}  ·  2×2, s=2`)}
      </div>
    </SimShell>
  );
}
