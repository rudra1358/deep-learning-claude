import type { Metadata } from "next";
import { Container, PageHeader } from "@/components/pages/PageHeader";
import { CodePlayground } from "@/components/pages/CodePlayground";

export const metadata: Metadata = { title: "Code Playground" };

export default function CodePage() {
  return (
    <Container wide>
      <PageHeader eyebrow="Code playground" title="Run Python in your browser">
        Edit and run NumPy, Matplotlib and scikit-learn code with a real CPython interpreter compiled to WebAssembly (Pyodide). Code never leaves your machine.
      </PageHeader>
      <CodePlayground />
    </Container>
  );
}
