"use client";

import { useEffect, useRef } from "react";
import { drawDetections, fitCanvas, type Drawable } from "@/lib/draw";
import { useElementSize } from "@/lib/hooks";

/**
 * An <img> with a canvas overlay of the same displayed size. Boxes are in original image
 * pixels (`imageWidth` x `imageHeight`, as reported by the API), so they are scaled by
 * displayed / original — this keeps them aligned at any screen width.
 */
export function DetectionImage({
  src,
  alt,
  items,
  imageWidth,
  imageHeight,
}: {
  src: string;
  alt: string;
  items: readonly Drawable[];
  imageWidth: number;
  imageHeight: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [wrapRef, size] = useElementSize<HTMLDivElement>();

  useEffect(() => {
    const c = canvasRef.current;
    if (!c || size.width === 0 || imageWidth === 0) return;
    const ctx = fitCanvas(c, size.width, size.height);
    if (!ctx) return;
    drawDetections(ctx, items, size.width / imageWidth, size.height / (imageHeight || 1));
  }, [items, size, imageWidth, imageHeight]);

  return (
    <div ref={wrapRef} className="relative w-full overflow-hidden rounded-md bg-black">
      {/* eslint-disable-next-line @next/next/no-img-element -- blob: URLs and API images, not optimisable */}
      <img src={src} alt={alt} className="block h-auto w-full" />
      <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden />
    </div>
  );
}
