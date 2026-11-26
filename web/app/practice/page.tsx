import type { Metadata } from "next";
import Link from "next/link";
import { LESSONS } from "@/content/lessons";
import { getTopic } from "@/content/curriculum";
import type { Block, QuizQuestion } from "@/lib/types";
import { Container, PageHeader } from "@/components/pages/PageHeader";
import { PracticeList } from "@/components/pages/PracticeList";

export const metadata: Metadata = { title: "Practice & Quizzes" };

function collect(blocks: Block[]): { quizzes: QuizQuestion[][]; exercises: number } {
  const quizzes: QuizQuestion[][] = [];
  let exercises = 0;
  for (const b of blocks) {
    if (b.t === "quiz") quizzes.push(b.questions);
    if (b.t === "exercise") exercises++;
  }
  return { quizzes, exercises };
}

export default function PracticePage() {
  const items = LESSONS.map((l) => {
    const found = getTopic(l.id)!;
    const { quizzes, exercises } = collect(l.sections.practice);
    const questions = quizzes.flat();
    const interview = l.sections.interview.filter((b) => b.t === "qa").length;
    return { id: l.id, title: l.title, module: `Module ${found.module.number} · ${found.module.title}`, questions: questions.length, exercises, interview, quizKey: `${l.id}:practice` };
  });
  return (
    <Container>
      <PageHeader eyebrow="Practice" title="Quizzes, exercises and interview questions">
        Every finished lesson ends with a knowledge check: multiple-choice and number questions, exercises with solutions, and interview questions sorted by level. Your best score on each quiz is saved.
      </PageHeader>
      <PracticeList items={items} />
      <p className="py-6 text-sm text-faint">
        Quizzes sit in each lesson&apos;s <em>Knowledge Check</em> section, so the material is right above if you need a refresher. <Link href="/progress" className="text-accent">See your scores →</Link>
      </p>
    </Container>
  );
}
