import { Component, type ReactNode } from "react";

interface Props {
  fallback: ReactNode;
  children: ReactNode;
}

/** Contains rendering failures (e.g. no WebGL) so the rest of the app keeps working. */
export default class ErrorBoundary extends Component<Props, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: unknown) {
    console.error("Rendering error contained by ErrorBoundary:", error);
  }

  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}
