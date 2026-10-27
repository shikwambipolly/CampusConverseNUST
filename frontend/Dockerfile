# =============================================================================
# frontend/Dockerfile — CampusConverse NUST Web UI Image (React + TypeScript)
# =============================================================================
# Stage 1: Compile TypeScript & bundle Vite React application
# Stage 2: Serve static assets via lightweight Nginx with /api reverse proxy
# =============================================================================

FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json ./
RUN npm install

COPY tsconfig.json vite.config.ts index.html ./
COPY src/ ./src/

RUN npm run build

# Stage 2: Production Nginx Server
FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1/ > /dev/null || exit 1

CMD ["nginx", "-g", "daemon off;"]
