import { defineConfig, loadEnv } from "vite";

export function resolveApiBase(env) {
  return env.VITE_API_BASE || "http://127.0.0.1:8000";
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiBase = resolveApiBase(env);

  return {
    server: {
      host: "0.0.0.0",
      port: 5173,
      proxy: {
        "/upload": apiBase,
        "/nlp": apiBase,
        "/train": apiBase,
        "/predict": apiBase,
        "/image": apiBase,
      },
    },
    preview: {
      host: "0.0.0.0",
      port: 4173,
    },
  };
});
