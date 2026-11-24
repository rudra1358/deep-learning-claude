import type { Metadata } from "next";
import Link from "next/link";
import { SimEmbed } from "@/components/sims/SimEmbed";

export const metadata: Metadata = { title: "Neural Network Visualizer" };

const HOW = [
  ["Design", "Add or remove hidden layers (up to four) and neurons, and choose each layer's activation and the loss."],
  ["Set data", "Type input values x and the target y, or click an input neuron and drag its slider."],
  ["Forward", "Press Step: each neuron computes its weighted sum z and activation a. The narration shows the arithmetic with the actual numbers."],
  ["Loss", "The output is compared to the target; ∂L/∂a starts the backward pass."],
  ["Backward", "Gradients δ = ∂L/∂z flow from output to input. Each step shows the chain-rule product and the weight gradients on the incoming edges."],
  ["Update", "Every parameter moves by −η·gradient. The panel lists old → new values and the loss before and after. Apply it, or tick “keep training”."],
  ["Verify", "Click any connection: the inspector compares the backprop gradient with a numerical central-difference gradient."],
];

export default function VisualizerPage() {
  return (
    <div className="mx-auto max-w-[1300px] px-4 py-8">
      <h1 className="font-serif text-3xl font-semibold">Neural Network Visualizer</h1>
      <p className="mt-2 max-w-3xl text-muted">
        A genuine simulation, not an animation: forward propagation, loss, backpropagation and gradient descent computed exactly, one neuron at a time. The full theory is in{" "}
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
