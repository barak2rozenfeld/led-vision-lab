import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const repoRoot = path.resolve(__dirname, "..");

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@samples": path.join(repoRoot, "data", "samples", "images"),
    },
  },
  server: {
    port: 5173,
    // Sample images live in data/samples, outside the web/ folder.
    fs: { allow: [repoRoot] },
    proxy: {
      "/api": {
        target: process.env.LED_API_URL ?? "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
