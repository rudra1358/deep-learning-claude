import type { Metadata } from "next";
import { Container, PageHeader } from "@/components/pages/PageHeader";
import { ArchitectureExplorer } from "@/components/pages/ArchitectureExplorer";

export const metadata: Metadata = { title: "Architecture Explorer" };

export default function ArchitecturesPage() {
  return (
    <Container>
      <PageHeader eyebrow="Architecture explorer" title="Shapes and sizes, layer by layer">
        Pick a famous network or build your own. You will see each layer's output shape and how many parameters (learnable numbers) it has. The totals match PyTorch exactly.
      </PageHeader>
      <ArchitectureExplorer />
    </Container>
  );
}
