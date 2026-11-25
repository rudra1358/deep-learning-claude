// Vite = the dev server + production bundler for the React app.
//   npm run dev    -> http://localhost:3100 with hot reload
//   npm run build  -> static files in dist/ (the Express server serves them in production)
//
// The proxy forwards /api (HTTP and the /api/stream WebSocket) to the Express server, so the
// browser sees a single origin and no CORS setup is needed.
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const API = process.env.API_PROXY_TARGET ?? "http://127.0.0.1:4100";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3100,
    proxy: { "/api": { target: API, ws: true } },
  },
  preview: {
    port: 3100,
    proxy: { "/api": { target: API, ws: true } },
  },
});
