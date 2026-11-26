import type { Metadata } from "next";
import { Container, PageHeader } from "@/components/pages/PageHeader";
import { ProgressDashboard } from "@/components/pages/ProgressDashboard";
import { LESSONS } from "@/content/lessons";
import { PROJECTS } from "@/content/projects";

export const metadata: Metadata = { title: "Progress Dashboard" };

export default function ProgressPage() {
  return (
    <Container>
      <PageHeader eyebrow="Your progress" title="Your learning so far" />
      <ProgressDashboard published={LESSONS.map((l) => l.id)} projects={PROJECTS.map((p) => ({ id: p.id, title: p.title, total: p.milestones.length }))} />
    </Container>
  );
}
