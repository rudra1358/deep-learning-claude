# Server image = the Express API + the built React app (one container serves both on port 4100).
# Build context: the project root (see docker-compose.yml).

# --- stage 1: build the React client into static files
FROM node:22-alpine AS client
WORKDIR /client
COPY client/package.json client/package-lock.json ./
RUN npm ci
COPY client/index.html client/vite.config.js ./
COPY client/src ./src
RUN npm run build

# --- stage 2: the Node server (production dependencies only) + the built client
FROM node:22-alpine
ENV NODE_ENV=production
WORKDIR /app/server
COPY server/package.json server/package-lock.json ./
RUN npm ci --omit=dev
COPY server/src ./src
COPY --from=client /client/dist /app/client/dist
RUN mkdir -p /app/data && chown -R node /app
USER node
ENV PORT=4100 DATA_DIR=/app/data CLIENT_DIST=/app/client/dist
EXPOSE 4100
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD wget -qO- http://localhost:4100/api/health || exit 1
CMD ["node", "src/index.js"]
