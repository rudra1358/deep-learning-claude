// MongoDB connection (Mongoose). One connection pool is shared by the whole process.
import mongoose from "mongoose";

export async function connectDb(url) {
  await mongoose.connect(url, { serverSelectionTimeoutMS: 5000 });
  return mongoose.connection;
}

export function dbState() {
  // 1 = connected (see mongoose.ConnectionStates)
  return mongoose.connection.readyState === 1 ? "ok" : "down";
}
