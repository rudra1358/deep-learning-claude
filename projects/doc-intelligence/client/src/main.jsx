// Entry point: mount React into <div id="root"> (index.html) and switch on client-side routing.
// BrowserRouter keeps the URL in sync with the page shown, without full page reloads.
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App.jsx";
import "./index.css";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
);
