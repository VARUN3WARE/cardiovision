import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
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
