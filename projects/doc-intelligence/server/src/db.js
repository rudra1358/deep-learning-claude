// MongoDB connection (through Mongoose) and the JSON shape every model uses in API responses.
import mongoose from "mongoose";

// Every model is sent to the browser with `id` (the ObjectId as a string) instead of Mongo's
// internal `_id` and `__v` fields - the same shape the old SQL API had, so the React code is simple.
export const jsonOptions = {
  virtuals: true, // adds `id`
  versionKey: false, // drops `__v`
  transform: (_doc, ret) => {
    delete ret._id;
    return ret;
  },
};

export async function connectDb(url) {
  await mongoose.connect(url, { serverSelectionTimeoutMS: 5000 });
  return mongoose.connection;
}

export const dbIsUp = () => mongoose.connection.readyState === 1; // 1 = connected

export const isValidId = (id) => mongoose.isValidObjectId(id) && String(id).length === 24;
