import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    // Optimize chunk sizes
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'ui-components': [
            './src/components/ui/button',
            './src/components/ui/card',
            './src/components/ui/badge',
            './src/components/ui/textarea',
          ],
        },
      },
    },
  },
  server: {
    host: '0.0.0.0', // Listen on all network interfaces for Runpod
    port: 5173,
    strictPort: true,
    // Improve dev server performance
    hmr: {
      overlay: false, // Disable error overlay for faster updates
      clientPort: 5173, // Use the same port for HMR
    },
    watch: {
      usePolling: false, // Disable polling for better performance
    },
  },
  optimizeDeps: {
    // Pre-bundle these dependencies for faster initial load
    include: ['react', 'react-dom', 'lucide-react'],
  },
})
