"use client";

/** Route-level error boundary: an unexpected render error shows this instead of a blank page. */
export default function Error({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <div role="alert" className="py-20 text-center">
      <h1 className="text-2xl font-semibold">Something went wrong</h1>
      <p className="mt-2 text-sm break-words text-muted">{error.message}</p>
      <button type="button" className="btn mt-6" onClick={reset}>
        Try again
      </button>
    </div>
  );
}
