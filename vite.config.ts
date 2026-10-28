import { defineConfig } from "vite";

// Tauri expects a fixed dev server port (matches src-tauri/tauri.conf.json's
// build.devUrl) and needs HMR to ignore its own Rust source tree.
export default defineConfig({
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    watch: {
      ignored: ["**/src-tauri/**", "**/orchestrator/**"],
    },
  },
});
