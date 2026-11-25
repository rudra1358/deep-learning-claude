// The shell around every page: <html>, the top navigation bar and the main content area.
// This is a Server Component (no "use client"): it renders once on the server and ships as HTML.
import type { Metadata } from "next";
import Nav from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "Document Intelligence",
  description: "Upload documents, search them and ask questions with cited answers.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <Nav />
        <main className="mx-auto w-full max-w-6xl px-4 py-6">{children}</main>
      </body>
    </html>
  );
}
