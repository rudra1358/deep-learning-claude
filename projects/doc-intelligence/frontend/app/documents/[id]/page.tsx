// Document viewer route: /documents/3?page=2&start=140&len=900&q=First+words+of+the+chunk
// This file is a Server Component: it only reads the URL and hands plain values to the
// client-side viewer, which does the fetching and scrolling in the browser.
import DocumentViewer from "@/components/DocumentViewer";

interface Props {
  params: Promise<{ id: string }>; // Next 15: params and searchParams are Promises
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}

// Read a query parameter as a number (or null when missing / not a number).
function num(v: string | string[] | undefined): number | null {
  const n = Number(Array.isArray(v) ? v[0] : v);
  return v === undefined || Number.isNaN(n) ? null : n;
}

export default async function DocumentPage({ params, searchParams }: Props) {
  const { id } = await params;
  const sp = await searchParams;
  const q = Array.isArray(sp.q) ? sp.q[0] : sp.q;

  return (
    <DocumentViewer
      id={Number(id)}
      target={{ page: num(sp.page), start: num(sp.start), len: num(sp.len), q: q ?? null }}
    />
  );
}
