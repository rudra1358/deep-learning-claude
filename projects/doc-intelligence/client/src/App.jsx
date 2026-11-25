// The shell around every page (top navigation bar + main content area) and the URL -> page table.
import { useEffect } from "react";
import { Link, Route, Routes, useLocation } from "react-router-dom";
import Nav from "./components/Nav.jsx";
import AskPage from "./pages/AskPage.jsx";
import DocumentPage from "./pages/DocumentPage.jsx";
import DocumentsPage from "./pages/DocumentsPage.jsx";
import MetricsPage from "./pages/MetricsPage.jsx";
import SearchPage from "./pages/SearchPage.jsx";

const TITLES = { "/": "Ask", "/documents": "Documents", "/search": "Search", "/metrics": "Metrics" };

// The browser tab title follows the page (there is no server rendering a <title> for us).
function useDocumentTitle() {
  const { pathname } = useLocation();
  useEffect(() => {
    const page = TITLES[pathname] ?? (pathname.startsWith("/documents/") ? "Document" : null);
    document.title = page ? `${page} · Document Intelligence` : "Document Intelligence";
  }, [pathname]);
}

function NotFound() {
  return (
    <div className="space-y-2">
      <h1 className="text-2xl font-semibold">Page not found</h1>
      <Link to="/" className="text-accent hover:underline">
        Back to Ask
      </Link>
    </div>
  );
}

export default function App() {
  useDocumentTitle();
  return (
    <>
      <Nav />
      <main className="mx-auto w-full max-w-6xl px-4 py-6">
        <Routes>
          <Route path="/" element={<AskPage />} />
          <Route path="/documents" element={<DocumentsPage />} />
          <Route path="/documents/:id" element={<DocumentPage />} />
          <Route path="/search" element={<SearchPage />} />
          <Route path="/metrics" element={<MetricsPage />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
    </>
  );
}
