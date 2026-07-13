# asset-service: Node/TypeScript + better-sqlite3 (native module, needs a
# toolchain to compile). Single build stage keeps node_modules/dist together
# so the already-compiled native binding travels with them into runtime.
FROM node:22-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends python3 make g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.json ./
COPY src ./src
RUN npm run build

FROM node:22-slim AS runtime
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY --from=build /app/package.json ./package.json
COPY drizzle ./drizzle

# db:migrate runs drizzle-orm's migrator against DATABASE_URL before the
# server starts -- matches the manual sequence used on the Pi deployment
# (apps/asset-service/README.md's quick start), just automated here.
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

EXPOSE 3002
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["node", "dist/index.js"]
