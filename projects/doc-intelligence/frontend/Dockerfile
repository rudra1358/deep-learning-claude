# syntax=docker/dockerfile:1
# Multi-stage build: 1) install deps  2) build  3) tiny runtime image with only the standalone server.
# Build:  docker build -t doc-intel-frontend --build-arg NEXT_PUBLIC_API_URL=http://localhost:8200 .
# Run:    docker run -p 3200:3200 doc-intel-frontend

FROM node:22-alpine AS deps
WORKDIR /app
RUN apk add --no-cache libc6-compat
COPY package.json package-lock.json ./
RUN npm ci

FROM node:22-alpine AS build
WORKDIR /app
# NEXT_PUBLIC_* values are inlined into the browser JavaScript at BUILD time, so the API URL is a
# build argument (changing it later means rebuilding). It is the URL the user's BROWSER calls.
ARG NEXT_PUBLIC_API_URL=http://localhost:8200
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL \
    NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3200 \
    HOSTNAME=0.0.0.0
# run as an unprivileged user, not root
RUN addgroup -S nodejs -g 1001 && adduser -S nextjs -u 1001 -G nodejs
COPY --from=build --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=build --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=build --chown=nextjs:nodejs /app/public ./public
USER nextjs
EXPOSE 3200
CMD ["node", "server.js"]
