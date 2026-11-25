// Server entry point:  npm start   (or npm run dev to restart on file changes)
//
// 1. connect to MongoDB   2. start HTTP + the live-stream WebSocket relay
// 3. resume watching video jobs that were running when the server last stopped
import http from "node:http";
import { createApp } from "./app.js";
import { config } from "./config.js";
import { connectDb } from "./db.js";
import { recoverJobs } from "./jobs.js";
import { attachStreamRelay } from "./stream.js";

await connectDb(config.mongoUrl);
const server = http.createServer(createApp());
attachStreamRelay(server);
server.listen(config.port, () => {
  console.log(`Vision Analytics server on http://localhost:${config.port}  (ML service: ${config.mlUrl})`);
});
const n = await recoverJobs();
if (n) console.log(`resumed watching ${n} video job(s)`);
