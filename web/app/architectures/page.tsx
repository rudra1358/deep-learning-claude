import type { Metadata } from "next";
import { Container, PageHeader } from "@/components/pages/PageHeader";
import { ArchitectureExplorer } from "@/components/pages/ArchitectureExplorer";

export const metadata: Metadata = { title: "Architecture Explorer" };

export default function ArchitecturesPage() {
  return (
    <Container>
      <PageHeader eyebrow="Architecture explorer" title="Shapes and parameters, layer by layer">
        Pick a classic architecture or build your own and see the output shape and parameter count of every layer, computed from the layer definitions. The totals match PyTorch exactly.
      </PageHeader>
      <ArchitectureExplorer />
    </Container>
  );
}
