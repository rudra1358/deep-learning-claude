import type { ReactNode } from "react";

export function PageHeader({ eyebrow, title, children }: { eyebrow?: string; title: string; children?: ReactNode }) {
  return (
    <header className="border-b border-border py-12">
      {eyebrow && <p className="mb-2 text-sm text-accent">{eyebrow}</p>}
      <h1 className="font-serif text-4xl font-semibold tracking-tight">{title}</h1>
      {children && <div className="mt-4 max-w-3xl text-lg text-muted">{children}</div>}
    </header>
  );
}

export function Container({ children, wide = false }: { children: ReactNode; wide?: boolean }) {
  return <div className={`mx-auto px-4 ${wide ? "max-w-[1400px]" : "max-w-[1200px]"}`}>{children}</div>;
}
