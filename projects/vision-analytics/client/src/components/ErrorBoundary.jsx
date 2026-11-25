import { Component } from "react";

/**
 * Error boundary: an unexpected render error shows this instead of a blank page.
 * (React only supports error boundaries as class components.)
 */
export class ErrorBoundary extends Component {
  state = { error: null };

  static getDerivedStateFromError(error) {
    return { error };
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <div role="alert" className="py-20 text-center">
        <h1 className="text-2xl font-semibold">Something went wrong</h1>
        <p className="mt-2 text-sm break-words text-muted">{error.message}</p>
        <button type="button" className="btn mt-6" onClick={() => this.setState({ error: null })}>
          Try again
        </button>
      </div>
    );
  }
}
