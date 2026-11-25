import { Link } from "react-router-dom";
import { usePageTitle } from "../lib/hooks";

export default function NotFound() {
  usePageTitle("Page not found");
  return (
    <div className="py-20 text-center">
      <h1 className="text-2xl font-semibold">Page not found</h1>
      <p className="mt-2 text-sm text-muted">The page or run you are looking for does not exist.</p>
      <Link to="/" className="btn mt-6">
        Back to overview
      </Link>
    </div>
  );
}
