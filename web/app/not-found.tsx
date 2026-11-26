import Link from "next/link";

export default function NotFound() {
  return (
    <div className="mx-auto max-w-xl px-4 py-32 text-center">
      <div className="font-mono text-sm text-faint">404</div>
      <h1 className="mt-2 font-serif text-3xl font-semibold">This page doesn&apos;t exist</h1>
      <p className="mt-3 text-muted">Looks like the gradient took a wrong turn. Try the lesson list, or search with Ctrl K.</p>
      <Link href="/curriculum" className="sim-btn primary mt-6">
        See all lessons
      </Link>
    </div>
  );
}
