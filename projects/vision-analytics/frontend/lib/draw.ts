/**
 * Canvas drawing shared by every overlay (image, live webcam, video playback, compare).
 *
 * Boxes from the API are in *original image pixels*. The canvas is sized to the element's
 * *displayed* size, so callers pass scaleX = displayedWidth / imageWidth (same for Y).
 */
import type { Box } from "./api";

export interface Drawable {
  box: Box;
  label: string;
  score: number;
  /** tracker id — shown as "#id" when present */
  id?: number;
  /** recent centre points in image pixels */
  trail?: [number, number][];
}

/**
 * Stable colour per label: hash the string (FNV-1a) to a hue, so "car" is always the same
 * colour on every page. Multiplying by the golden angle spreads similar hashes around the wheel.
 */
export function colorForLabel(label: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < label.length; i++) {
    h ^= label.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return `hsl(${Math.round((h * 137.508) % 360)} 80% 50%)`;
}

/** Resize a canvas's backing store to its CSS size × devicePixelRatio (sharp on HiDPI). Returns the 2D context in CSS pixels. */
export function fitCanvas(canvas: HTMLCanvasElement, cssWidth: number, cssHeight: number): CanvasRenderingContext2D | null {
  const dpr = typeof window === "undefined" ? 1 : window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(cssWidth * dpr));
  const h = Math.max(1, Math.round(cssHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  return ctx;
}

export function drawDetections(
  ctx: CanvasRenderingContext2D,
  items: readonly Drawable[],
  scaleX: number,
  scaleY: number,
): void {
  ctx.save();
  ctx.font = "600 12px ui-sans-serif, system-ui, sans-serif";
  ctx.textBaseline = "top";
  ctx.lineJoin = "round";

  for (const it of items) {
    const [x1, y1, x2, y2] = it.box;
    const x = x1 * scaleX;
    const y = y1 * scaleY;
    const w = (x2 - x1) * scaleX;
    const h = (y2 - y1) * scaleY;
    const color = colorForLabel(it.label);

    // motion trail (oldest point most transparent)
    if (it.trail && it.trail.length > 1) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      for (let i = 1; i < it.trail.length; i++) {
        const a = it.trail[i - 1];
        const b = it.trail[i];
        if (!a || !b) continue;
        ctx.globalAlpha = 0.15 + (0.85 * i) / it.trail.length;
        ctx.beginPath();
        ctx.moveTo(a[0] * scaleX, a[1] * scaleY);
        ctx.lineTo(b[0] * scaleX, b[1] * scaleY);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    const text = `${it.id !== undefined ? `#${it.id} ` : ""}${it.label} ${it.score.toFixed(2)}`;
    const tw = ctx.measureText(text).width + 8;
    const th = 16;
    const ty = y - th >= 0 ? y - th : y; // put the label inside the box if it would leave the canvas
    ctx.fillStyle = color;
    ctx.fillRect(x - 1, ty, tw, th);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, x + 3, ty + 2);
  }
  ctx.restore();
}

/** Draw a counting line given in normalised 0..1 coords onto a canvas of the given CSS size. */
export function drawCountingLine(
  ctx: CanvasRenderingContext2D,
  line: readonly [number, number, number, number],
  width: number,
  height: number,
): void {
  const [x1, y1, x2, y2] = line;
  ctx.save();
  ctx.strokeStyle = "#facc15";
  ctx.lineWidth = 3;
  ctx.setLineDash([10, 6]);
  ctx.beginPath();
  ctx.moveTo(x1 * width, y1 * height);
  ctx.lineTo(x2 * width, y2 * height);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#facc15";
  for (const [px, py] of [
    [x1, y1],
    [x2, y2],
  ] as const) {
    ctx.beginPath();
    ctx.arc(px * width, py * height, 5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}
