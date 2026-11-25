// Entry point:  npm start   (or npm run dev to restart on file changes)
// 1) connect to MongoDB  2) re-queue jobs a previous run left unfinished  3) start listening.
import { createApp } from "./app.js";
import { config } from "./config.js";
import { connectDb } from "./db.js";
import { recover } from "./jobs.js";

await connectDb(config.mongoUrl);
console.log(`MongoDB connected: ${config.mongoUrl}`);

const n = await recover();
if (n) console.log(`re-queued ${n} interrupted document(s)`);

createApp().listen(config.port, () => {
  console.log(`API on http://localhost:${config.port}/api  (ML service: ${config.mlUrl})`);
});
