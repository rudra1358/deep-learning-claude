"use client"; // needs usePathname (a browser-side hook) to highlight the current page

import Link from "next/link";
import { usePathname } from "next/navigation";
import HealthBadge from "./HealthBadge";

const LINKS = [
  { href: "/", label: "Ask" },
  { href: "/documents", label: "Documents" },
  { href: "/search", label: "Search" },
  { href: "/metrics", label: "Metrics" },
];

export default function Nav() {
  const pathname = usePathname();

  // "/documents/3" should still highlight "Documents"; "/" must match exactly or it would match everything.
  const isActive = (href: string) => (href === "/" ? pathname === "/" : pathname.startsWith(href));

  return (
    <header className="border-b border-border bg-card">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
        <Link href="/" className="font-semibold">
          Doc Intelligence
        </Link>
        <nav aria-label="Main" className="flex flex-wrap gap-1 text-sm">
          {LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              aria-current={isActive(l.href) ? "page" : undefined}
              className={`rounded px-3 py-1.5 ${
                isActive(l.href) ? "bg-accent text-accent-fg" : "text-muted hover:text-fg"
              }`}
            >
              {l.label}
            </Link>
          ))}
        </nav>
        <div className="ml-auto">
          <HealthBadge />
        </div>
      </div>
    </header>
  );
}
