import { defineConfig } from "vite";

export default defineConfig({
  server: {
    proxy: {
      "/upload": "http://127.0.0.1:8001",
      "/nlp": "http://127.0.0.1:8001",
      "/train": "http://127.0.0.1:8001",
      "/predict": "http://127.0.0.1:8001",
    },
  },
});
