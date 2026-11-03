import { defineConfig } from "vite";

// Dev server proxies the JSON API to the Python backend (python -m aegis.server).
export default defineConfig({
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8765" } },
  build: { outDir: "dist", emptyOutDir: true, target: "es2020" },
});
