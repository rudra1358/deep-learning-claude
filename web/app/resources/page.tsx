import type { Metadata } from "next";
import { ExternalLink } from "lucide-react";
import { Container, PageHeader } from "@/components/pages/PageHeader";

export const metadata: Metadata = { title: "Resources" };

const GROUPS: { title: string; items: { name: string; by: string; url: string; note: string }[] }[] = [
  {
    title: "Textbooks (free online)",
    items: [
      { name: "Deep Learning", by: "Goodfellow, Bengio & Courville (MIT Press, 2016)", url: "https://www.deeplearningbook.org/", note: "The standard graduate reference; strong on mathematical foundations." },
      { name: "Dive into Deep Learning", by: "Zhang, Lipton, Li & Smola", url: "https://d2l.ai/", note: "Interactive book with runnable PyTorch/TensorFlow/JAX code." },
      { name: "Mathematics for Machine Learning", by: "Deisenroth, Faisal & Ong (Cambridge, 2020)", url: "https://mml-book.github.io/", note: "Linear algebra, calculus and probability for ML." },
      { name: "Neural Networks and Deep Learning", by: "Michael Nielsen", url: "http://neuralnetworksanddeeplearning.com/", note: "A gentle, thorough derivation of backpropagation." },
    ],
  },
  {
    title: "Official documentation",
    items: [
      { name: "PyTorch documentation & tutorials", by: "PyTorch", url: "https://pytorch.org/docs/stable/index.html", note: "API reference for torch, nn, optim, autograd." },
      { name: "TensorFlow / Keras guides", by: "TensorFlow", url: "https://www.tensorflow.org/guide", note: "Keras APIs, GradientTape, SavedModel." },
      { name: "NumPy documentation", by: "NumPy", url: "https://numpy.org/doc/stable/", note: "Arrays, broadcasting, linear algebra." },
      { name: "Hugging Face documentation", by: "Hugging Face", url: "https://huggingface.co/docs", note: "Transformers, Tokenizers, Diffusers, Datasets." },
    ],
  },
  {
    title: "Courses",
    items: [
      { name: "CS231n: Deep Learning for Computer Vision", by: "Stanford", url: "https://cs231n.stanford.edu/", note: "Excellent notes on CNNs and backpropagation." },
      { name: "CS224n: Natural Language Processing with Deep Learning", by: "Stanford", url: "https://web.stanford.edu/class/cs224n/", note: "Word vectors through Transformers and LLMs." },
    ],
  },
  {
    title: "Datasets",
    items: [
      { name: "Hugging Face Datasets Hub", by: "Hugging Face", url: "https://huggingface.co/datasets", note: "Thousands of text, vision and audio datasets." },
      { name: "torchvision datasets", by: "PyTorch", url: "https://pytorch.org/vision/stable/datasets.html", note: "MNIST, CIFAR, ImageNet loaders and more." },
      { name: "COCO", by: "Common Objects in Context", url: "https://cocodataset.org/", note: "Detection, segmentation and captioning benchmark." },
    ],
  },
];

export default function ResourcesPage() {
  return (
    <Container>
      <PageHeader eyebrow="Resources" title="Further reading and tools">
        A short, curated list. Use these alongside the lessons — not instead of understanding the derivations yourself.
      </PageHeader>
      {GROUPS.map((g) => (
        <section key={g.title} className="py-8">
          <h2 className="mb-4 font-serif text-2xl font-semibold">{g.title}</h2>
          <ul className="grid gap-3 md:grid-cols-2">
            {g.items.map((it) => (
              <li key={it.url}>
                <a href={it.url} target="_blank" rel="noreferrer" className="card block h-full p-4 hover:border-border-strong">
                  <div className="flex items-center gap-1.5 font-medium">
                    {it.name} <ExternalLink size={12} className="text-faint" />
                  </div>
                  <div className="text-xs text-faint">{it.by}</div>
                  <p className="mt-1 text-sm text-muted">{it.note}</p>
                </a>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </Container>
  );
}
