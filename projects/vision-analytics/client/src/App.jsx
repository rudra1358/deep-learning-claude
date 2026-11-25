// Page layout (navigation bar + main area) and the URL -> page mapping.
import { Route, Routes, useLocation } from "react-router-dom";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { Nav } from "./components/Nav";
import Compare from "./pages/Compare";
import History from "./pages/History";
import Image from "./pages/Image";
import Live from "./pages/Live";
import Metrics from "./pages/Metrics";
import NotFound from "./pages/NotFound";
import Overview from "./pages/Overview";
import RunPage from "./pages/RunPage";
import Video from "./pages/Video";

export default function App() {
  const { pathname } = useLocation();
  return (
    <>
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 btn">
        Skip to content
      </a>
      <Nav />
      <main id="main" className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6">
        {/* key: a crash on one page is forgotten when you navigate to another */}
        <ErrorBoundary key={pathname}>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/live" element={<Live />} />
            <Route path="/image" element={<Image />} />
            <Route path="/video" element={<Video />} />
            <Route path="/history" element={<History />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/metrics" element={<Metrics />} />
            <Route path="/runs/:id" element={<RunPage />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </ErrorBoundary>
      </main>
    </>
  );
}
