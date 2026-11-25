// Top navigation bar. NavLink knows the current URL and marks the matching link as active.
import { Link, NavLink } from "react-router-dom";
import HealthBadge from "./HealthBadge.jsx";

const LINKS = [
  { to: "/", label: "Ask" },
  { to: "/documents", label: "Documents" },
  { to: "/search", label: "Search" },
  { to: "/metrics", label: "Metrics" },
];

export default function Nav() {
  return (
    <header className="border-b border-border bg-card">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
        <Link to="/" className="font-semibold">
          Doc Intelligence
        </Link>
        <nav aria-label="Main" className="flex flex-wrap gap-1 text-sm">
          {LINKS.map((l) => (
            // "/documents/3" still highlights "Documents"; `end` makes "/" match exactly, or it would match everything.
            // NavLink sets aria-current="page" on the active link by itself.
            <NavLink
              key={l.to}
              to={l.to}
              end={l.to === "/"}
              className={({ isActive }) =>
                `rounded px-3 py-1.5 ${isActive ? "bg-accent text-accent-fg" : "text-muted hover:text-fg"}`
              }
            >
              {l.label}
            </NavLink>
          ))}
        </nav>
        <div className="ml-auto">
          <HealthBadge />
        </div>
      </div>
    </header>
  );
}
