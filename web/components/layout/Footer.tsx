import Link from "next/link";

const COLS = [
  { title: "Learn", links: [["Curriculum", "/curriculum"], ["Roadmap", "/roadmap"], ["Math Lab", "/math-lab"], ["Practice", "/practice"]] },
  { title: "Explore", links: [["Playground", "/playground"], ["Network Visualizer", "/visualizer"], ["Architectures", "/architectures"], ["Code Playground", "/code"]] },
  { title: "Reference", links: [["Research Papers", "/papers"], ["Glossary", "/glossary"], ["Resources", "/resources"], ["Projects", "/projects"]] },
];

export function Footer() {
  return (
    <footer className="mt-24 border-t border-border">
      <div className="mx-auto grid max-w-[1200px] gap-8 px-4 py-12 sm:grid-cols-2 lg:grid-cols-4">
        <div>
          <div className="font-serif font-semibold">Deep Learning University</div>
          <p className="mt-2 text-sm text-muted">Learn deep learning by watching it work — every number in every simulator is really calculated.</p>
        </div>
        {COLS.map((c) => (
          <div key={c.title}>
            <div className="mb-2 text-xs font-medium uppercase tracking-wider text-faint">{c.title}</div>
            <ul className="space-y-1.5 text-sm">
              {c.links.map(([l, h]) => (
                <li key={h}>
                  <Link href={h} className="text-muted hover:text-text">
                    {l}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </footer>
  );
}
