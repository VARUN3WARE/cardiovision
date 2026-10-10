/**
 * Runs `cb` when the browser is next idle, or after `timeout` ms at the latest (a busy main thread, e.g. slow
 * software rendering, has few idle periods). Returns a cancel function.
 */
export function whenIdle(cb: (deadline: { timeRemaining: () => number }) => void, timeout: number): () => void {
  if (typeof window.requestIdleCallback === "function") {
    const id = window.requestIdleCallback(cb, { timeout });
    return () => window.cancelIdleCallback(id);
  }
  const id = window.setTimeout(() => cb({ timeRemaining: () => 8 }), Math.min(timeout, 200));
  return () => window.clearTimeout(id);
}
