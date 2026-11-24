import { forwardBackward, initNet } from "@/lib/sim/nn";

/**
 * Hero illustration — not decoration: a real 3-4-2 network evaluated with the same
 * engine as the visualizer. Values on neurons are actual activations; numbers in
 * clay under the neurons are actual backpropagated gradients δ = ∂L/∂z.
 */
export function HeroNetwork() {
  const net = initNet([3, 4, 2], ["tanh", "sigmoid"], 11);
  const x = [0.8, -0.4, 0.3];
  const y = [1, 0];
  const tr = forwardBackward(net, x, y, "bce");
  const W = 420, H = 300, R = 19;
  const colX = (l: number) => 50 + l * 160;
  const rowY = (l: number, j: number) => H / 2 + (j - (net.sizes[l] - 1) / 2) * 62;
  const maxW = Math.max(...net.W.flat(2).map(Math.abs));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="A small neural network showing real activations and gradients">
      {net.W.map((Wl, l) =>
        Wl.map((row, j) =>
          row.map((w, k) => (
            <line
              key={`${l}${j}${k}`}
              x1={colX(l) + R}
              y1={rowY(l, k)}
              x2={colX(l + 1) - R}
              y2={rowY(l + 1, j)}
              stroke={w >= 0 ? "var(--accent)" : "var(--clay)"}
              strokeOpacity={0.25 + (0.6 * Math.abs(w)) / maxW}
              strokeWidth={0.8 + (1.6 * Math.abs(w)) / maxW}
            />
          )),
        ),
      )}
      {net.sizes.map((n, l) =>
        Array.from({ length: n }, (_, j) => (
          <g key={`${l}${j}`}>
            <circle cx={colX(l)} cy={rowY(l, j)} r={R} fill="var(--surface)" stroke="var(--border-strong)" />
            <text x={colX(l)} y={rowY(l, j) + 4} textAnchor="middle" fontSize={10.5} fontFamily="var(--font-mono)" fill="var(--text)">
              {tr.a[l][j].toFixed(2)}
            </text>
            {l > 0 && (
              <text x={colX(l)} y={rowY(l, j) + R + 12} textAnchor="middle" fontSize={9} fontFamily="var(--font-mono)" fill="var(--clay)">
                δ {tr.delta[l][j].toFixed(3)}
              </text>
            )}
          </g>
        )),
      )}
      <text x={colX(0)} y={24} textAnchor="middle" fontSize={10.5} fill="var(--text-faint)">input</text>
      <text x={colX(1)} y={24} textAnchor="middle" fontSize={10.5} fill="var(--text-faint)">hidden · tanh</text>
      <text x={colX(2)} y={24} textAnchor="middle" fontSize={10.5} fill="var(--text-faint)">output · σ</text>
      <text x={W - 8} y={H - 8} textAnchor="end" fontSize={10.5} fontFamily="var(--font-mono)" fill="var(--text-muted)">
        BCE loss = {tr.loss.toFixed(4)}
      </text>
    </svg>
  );
}
