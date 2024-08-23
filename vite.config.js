// Set host to true to allow access from other devices on the same network

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  server: {
    watch: {
      usePolling: true,
    },
    host: true, // Here
  },
  plugins: [react()],
});
