"use client";

import { CheckSquare, Square } from "lucide-react";
import { progress, useProgress } from "@/lib/progress";

export function MilestoneList({ projectId, milestones }: { projectId: string; milestones: { id: string; title: string; detail: string }[] }) {
  const p = useProgress();
  const done = p.projects[projectId] ?? [];
  return (
    <ul className="space-y-1">
      {milestones.map((m) => {
        const ok = done.includes(m.id);
        return (
          <li key={m.id}>
            <button onClick={() => progress.toggleMilestone(projectId, m.id)} className="flex w-full gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-surface-2" aria-pressed={ok}>
              {ok ? <CheckSquare size={16} className="mt-0.5 shrink-0 text-accent" /> : <Square size={16} className="mt-0.5 shrink-0 text-faint" />}
              <span>
                <span className="font-medium">{m.title}</span> <span className="text-muted">— {m.detail}</span>
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
