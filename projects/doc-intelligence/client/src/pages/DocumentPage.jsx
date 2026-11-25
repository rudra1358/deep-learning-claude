// Document viewer route: /documents/<id>?page=2&start=140&len=900&q=First+words+of+the+chunk
// This page only reads the URL and hands plain values to the viewer component, which does the
// fetching and scrolling.
import { useParams, useSearchParams } from "react-router-dom";
import DocumentViewer from "../components/DocumentViewer.jsx";

// Read a query parameter as a number (or null when missing / not a number).
function num(v) {
  const n = Number(v);
  return v === null || Number.isNaN(n) ? null : n;
}

export default function DocumentPage() {
  const { id } = useParams(); // a MongoDB ObjectId string
  const [sp] = useSearchParams();

  return (
    <DocumentViewer
      id={id}
      target={{ page: num(sp.get("page")), start: num(sp.get("start")), len: num(sp.get("len")), q: sp.get("q") }}
    />
  );
}
