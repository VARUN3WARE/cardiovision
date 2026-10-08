import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // One copy of three for the app, drei, postprocessing and three-mesh-bvh (instanceof checks).
  resolve: { dedupe: ["three"] },
  optimizeDeps: { include: ["three", "three-mesh-bvh"] },
  server: {
    port: 5173,
    proxy: {
      "/predict": "http://localhost:8000",
      "/explain": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/features": "http://localhost:8000",
      "/docs": "http://localhost:8000",
      "/openapi.json": "http://localhost:8000",
      "/metrics": "http://localhost:8000",
    },
  },
});
