# syntax=docker/dockerfile:1
# The Node image: builds the React client, then runs the Express server, which serves the API
# (/api/...) AND the built UI from one port. Build context = the project root (needs client/ + server/).
#   docker build -f server/Dockerfile -t doc-intel-server .

# 1) build the React app into static files (client/dist)
FROM node:22-alpine AS client
WORKDIR /app/client
COPY client/package.json client/package-lock.json ./
RUN npm ci
COPY client/ ./
RUN npm run build

# 2) the server with production dependencies only
FROM node:22-alpine
ENV NODE_ENV=production PORT=4200 DATA_DIR=/app/data CLIENT_DIST=/app/client/dist
WORKDIR /app/server
COPY server/package.json server/package-lock.json ./
RUN npm ci --omit=dev
COPY server/src ./src
COPY --from=client /app/client/dist /app/client/dist
# run as an unprivileged user, not root; /app/data is the volume shared with the ML service
RUN mkdir -p /app/data && chown -R node /app/data
USER node
EXPOSE 4200
CMD ["node", "src/index.js"]
