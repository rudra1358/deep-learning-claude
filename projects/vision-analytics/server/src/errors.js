// Errors and the error-handling middleware.
//
// Every error the API returns has the shape {"detail": "message"} (the same shape the Python
// service uses), so the React client handles errors from both the same way.
import multer from "multer";

export class HttpError extends Error {
  constructor(status, detail) {
    super(detail);
    this.status = status;
  }
}

// Express 5 forwards errors thrown in async route handlers here automatically.
// eslint-disable-next-line no-unused-vars
export function errorHandler(err, req, res, next) {
  if (err instanceof multer.MulterError) {
    const status = err.code === "LIMIT_FILE_SIZE" ? 413 : 400;
    return res.status(status).json({ detail: status === 413 ? "file too large" : err.message });
  }
  if (err instanceof HttpError) {
    return res.status(err.status).json({ detail: err.message });
  }
  if (err?.name === "CastError") {
    // e.g. /api/runs/not-an-object-id
    return res.status(404).json({ detail: "not found" });
  }
  console.error(err);
  res.status(500).json({ detail: "internal server error" });
}
