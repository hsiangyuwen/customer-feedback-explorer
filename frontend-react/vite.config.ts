import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000, // Specify the development server port
    strictPort: true, // Fail if port 3000 is already in use
    // Optional: Proxy API requests to backend during development
    // This avoids CORS issues if backend is on a different port (e.g., 8000)
    // and allows using relative paths for API calls in your React app (e.g., /api/ingest)
    // proxy: {
    //   '/api': { // All requests to /api/* will be proxied
    //     target: 'http://localhost:8000', // Your backend API
    //     changeOrigin: true,
    //     rewrite: (path) => path.replace(/^\/api/, ''), // Remove /api prefix before sending to backend
    //   },
    // },
  },
})
