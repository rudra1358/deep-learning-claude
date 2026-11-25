// Vite = the dev server and bundler for the React app.
// In development the browser loads the UI from :3200 and calls /api/... on the SAME origin;
// the proxy below forwards those calls to the Express server on :4200 (so no CORS is needed).
// In production `npm run build` writes static files to dist/, which the Express server serves itself.
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3200,
    proxy: { "/api": "http://127.0.0.1:4200" },
  },
});
