import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { RunDetail } from "@/components/RunDetail";

export async function generateMetadata({ params }: { params: Promise<{ id: string }> }): Promise<Metadata> {
  const { id } = await params;
  return { title: `Run #${id}` };
}

/** Server wrapper: validates the id from the URL, then hands over to the client component that fetches the data. */
export default async function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const n = Number(id);
  if (!Number.isInteger(n) || n <= 0) notFound();
  return <RunDetail id={n} />;
}
