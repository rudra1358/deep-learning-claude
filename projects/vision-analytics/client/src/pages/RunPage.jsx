import { useParams } from "react-router-dom";
import { RunDetail } from "../components/RunDetail";
import { shortId } from "../lib/format";
import { usePageTitle } from "../lib/hooks";
import NotFound from "./NotFound";

/** /runs/:id - checks the id from the URL, then hands over to RunDetail, which fetches the data. */
export default function RunPage() {
  const { id } = useParams();
  const valid = /^[0-9a-f]{24}$/i.test(id ?? ""); // a MongoDB ObjectId
  usePageTitle(valid ? `Run #${shortId(id)}` : "Page not found");
  if (!valid) return <NotFound />;
  return <RunDetail id={id} />;
}
