# asset-service: Node/TypeScript + better-sqlite3 (native module) + a
# `file:` dependency on ../../infra/kms-adapter (envelope encryption at
# rest, src/crypto/imageEncryption.ts) -- build context is the repo root,
# not apps/asset-service alone, so that dependency actually resolves
# inside the image too (mirrors the real directory structure: /infra,
# /apps/asset-service -- same reasoning as apps/api-gateway/Dockerfile).
FROM node:22-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends python3 make g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /infra/kms-adapter
COPY infra/kms-adapter/package.json infra/kms-adapter/package-lock.json ./
RUN npm ci
COPY infra/kms-adapter/tsconfig.json ./
COPY infra/kms-adapter/src ./src
RUN npm run build

WORKDIR /apps/asset-service
COPY apps/asset-service/package.json apps/asset-service/package-lock.json ./
RUN npm ci
COPY apps/asset-service/tsconfig.json ./
COPY apps/asset-service/src ./src
RUN npm run build

FROM node:22-slim AS runtime
WORKDIR /apps/asset-service
ENV NODE_ENV=production
# node_modules/@dontai/kms-adapter is a symlink to /infra/kms-adapter --
# copying node_modules alone would leave it dangling in this stage.
COPY --from=build /infra/kms-adapter /infra/kms-adapter
COPY --from=build /apps/asset-service/node_modules ./node_modules
COPY --from=build /apps/asset-service/dist ./dist
COPY --from=build /apps/asset-service/package.json ./package.json
COPY apps/asset-service/drizzle ./drizzle
COPY apps/asset-service/kms-keys ./kms-keys

# db:migrate runs drizzle-orm's migrator against DATABASE_URL before the
# server starts -- matches the manual sequence used on the Pi deployment
# (apps/asset-service/README.md's quick start), just automated here.
COPY apps/asset-service/docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

EXPOSE 3002
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["node", "dist/index.js"]
