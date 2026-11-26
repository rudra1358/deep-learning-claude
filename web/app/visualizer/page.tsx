import type { Metadata } from "next";
import Link from "next/link";
import { SimEmbed } from "@/components/sims/SimEmbed";

export const metadata: Metadata = { title: "Neural Network Visualizer" };

const HOW = [
  ["Design", "Add or remove hidden layers (up to four) and neurons. Pick each layer's activation and the loss."],
  ["Set data", "Type input values x and the target y, or click an input neuron and drag its slider."],
  ["Forward", "Press Step: each neuron works out its weighted sum z and its output a. The notes show the math with the real numbers."],
  ["Loss", "The output is compared with the target. That error, ∂L/∂a, starts the backward pass."],
  ["Backward", "Gradients δ = ∂L/∂z flow back from output to input. Each step shows the chain-rule multiplication and the gradient for each incoming weight."],
  ["Update", "Every parameter moves by −η·gradient. The panel shows old → new values and the loss before and after. Apply it, or tick “keep training”."],
  ["Verify", "Click any connection to check it: its backprop gradient is compared with one estimated numerically (central difference)."],
];

export default function VisualizerPage() {
  return (
    <div className="mx-auto max-w-[1300px] px-4 py-8">
      <h1 className="font-serif text-3xl font-semibold">Neural Network Visualizer</h1>
      <p className="mt-2 max-w-3xl text-muted">
        A real simulation, not an animation: the forward pass, loss, backpropagation and gradient descent are computed exactly, one neuron at a time. The full explanation is in{" "}
        <Link href="/learn/m04/forward-propagation" className="text-accent underline underline-offset-2">
          Forward Propagation
        </Link>{" "}
        and{" "}
        <Link href="/learn/m04/backpropagation" className="text-accent underline underline-offset-2">
          Backpropagation
        </Link>
        .
      </p>
      <SimEmbed id="network" />
      <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {HOW.map(([t, d], i) => (
          <li key={t} className="card p-4 text-sm">
            <div className="font-mono text-xs text-faint">{i + 1}</div>
            <div className="font-medium">{t}</div>
            <p className="mt-1 text-muted">{d}</p>
          </li>
        ))}
      </ol>
    </div>
  );
}
