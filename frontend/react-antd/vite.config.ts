import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  },
  server: {
    port: 5173, // 使用 Vite 默认端口，避免 Windows 端口权限问题
    host: 'localhost', // 使用 localhost 而不是 127.0.0.1
    strictPort: false, // 如果端口被占用，自动尝试下一个端口
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})
