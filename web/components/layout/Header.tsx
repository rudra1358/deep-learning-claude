"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Menu, Moon, Search, Sun, X, BarChart3 } from "lucide-react";
import { useTheme } from "./ThemeProvider";
import { SearchDialog } from "./SearchDialog";

const NAV = [
  { href: "/curriculum", label: "Curriculum" },
  { href: "/roadmap", label: "Roadmap" },
  { href: "/playground", label: "Playground" },
  { href: "/math-lab", label: "Math Lab" },
  { href: "/architectures", label: "Architectures" },
  { href: "/code", label: "Code" },
  { href: "/practice", label: "Practice" },
  { href: "/projects", label: "Projects" },
];

const MORE = [
  { href: "/visualizer", label: "Neural Network Visualizer" },
  { href: "/papers", label: "Research Papers" },
  { href: "/glossary", label: "Glossary" },
  { href: "/resources", label: "Resources" },
  { href: "/progress", label: "Progress" },
];

export function Logo() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden>
      <g stroke="var(--accent)" strokeWidth="1.4" fill="none">
        <line x1="5" y1="6" x2="12" y2="12" />
        <line x1="5" y1="18" x2="12" y2="12" />
        <line x1="12" y1="12" x2="19" y2="7" />
        <line x1="12" y1="12" x2="19" y2="17" />
      </g>
      <circle cx="5" cy="6" r="2.4" fill="var(--accent)" />
      <circle cx="5" cy="18" r="2.4" fill="var(--accent)" />
      <circle cx="12" cy="12" r="2.8" fill="var(--text)" />
      <circle cx="19" cy="7" r="2.4" fill="var(--clay)" />
      <circle cx="19" cy="17" r="2.4" fill="var(--clay)" />
    </svg>
  );
}

export function Header() {
  const { theme, toggle } = useTheme();
  const path = usePathname();
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState(false);

  useEffect(() => setOpen(false), [path]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setSearch(true);
      }
      if (e.key === "/" && !(e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement)) {
        e.preventDefault();
        setSearch(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <header className="sticky top-0 z-40 border-b border-border bg-bg/90 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-[1400px] items-center gap-4 px-4">
        <Link href="/" className="flex items-center gap-2 font-serif text-[1.05rem] font-semibold">
          <Logo />
          <span className="hidden sm:inline">Deep Learning University</span>
          <span className="sm:hidden">DLU</span>
        </Link>
        <nav className="ml-4 hidden items-center gap-0.5 xl:flex" aria-label="Main">
          {NAV.map((n) => (
            <Link key={n.href} href={n.href} className={`rounded-md px-2.5 py-1.5 text-sm transition-colors ${path.startsWith(n.href) ? "text-text" : "text-muted hover:text-text"}`}>
              {n.label}
            </Link>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-1">
          <button onClick={() => setSearch(true)} className="flex items-center gap-2 rounded-lg border border-border bg-surface px-2.5 py-1.5 text-sm text-faint hover:text-muted" aria-label="Search the curriculum">
            <Search size={15} />
            <span className="hidden md:inline">Search</span>
            <kbd className="hidden rounded border border-border px-1 font-mono text-[0.65rem] md:inline">Ctrl K</kbd>
          </button>
          <Link href="/progress" className="rounded-md p-2 text-muted hover:text-text" aria-label="Your progress">
            <BarChart3 size={17} />
          </Link>
          <button onClick={toggle} className="rounded-md p-2 text-muted hover:text-text" aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}>
            {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
          </button>
          <button onClick={() => setOpen((o) => !o)} className="rounded-md p-2 text-muted hover:text-text xl:hidden" aria-label="Open menu" aria-expanded={open}>
            {open ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
      </div>
      {open && (
        <nav className="border-t border-border bg-bg px-4 py-3 xl:hidden" aria-label="Mobile">
          <div className="grid grid-cols-2 gap-1 sm:grid-cols-3">
            {[...NAV, ...MORE].map((n) => (
              <Link key={n.href} href={n.href} className="rounded-md px-2 py-2 text-sm text-muted hover:bg-surface-2 hover:text-text">
                {n.label}
              </Link>
            ))}
          </div>
        </nav>
      )}
      {search && <SearchDialog onClose={() => setSearch(false)} />}
    </header>
  );
}
