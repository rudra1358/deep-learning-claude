import type { Metadata } from "next";
import { Container, PageHeader } from "@/components/pages/PageHeader";
import { CodePlayground } from "@/components/pages/CodePlayground";

export const metadata: Metadata = { title: "Code Playground" };

export default function CodePage() {
  return (
    <Container wide>
      <PageHeader eyebrow="Code playground" title="Run Python in your browser">
        Edit and run NumPy, Matplotlib and scikit-learn code. Real Python runs right in your browser (using Pyodide), so your code never leaves your computer.
      </PageHeader>
      <CodePlayground />
    </Container>
  );
}
