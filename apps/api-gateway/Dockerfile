# api-gateway: Node/TypeScript + better-sqlite3 (native module) + a `file:`
# dependency on ../../infra/kms-adapter -- the reason this Dockerfile's
# build context is the repo root (see docker-compose.yml), not
# apps/api-gateway alone: the file: reference needs infra/kms-adapter
# actually present at that relative path *inside the image* too, so this
# mirrors the repo's real directory layout (/infra, /apps/api-gateway)
# rather than flattening it.
FROM node:22-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends python3 make g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /infra/kms-adapter
COPY infra/kms-adapter/package.json infra/kms-adapter/package-lock.json ./
RUN npm ci
COPY infra/kms-adapter/tsconfig.json ./
COPY infra/kms-adapter/src ./src
RUN npm run build

WORKDIR /apps/api-gateway
COPY apps/api-gateway/package.json apps/api-gateway/package-lock.json ./
RUN npm ci
COPY apps/api-gateway/tsconfig.json ./
COPY apps/api-gateway/src ./src
RUN npm run build

FROM node:22-slim AS runtime
WORKDIR /apps/api-gateway
ENV NODE_ENV=production
# node_modules/@dontai/kms-adapter is a symlink npm created pointing at
# /infra/kms-adapter (its `file:../../infra/kms-adapter` target) --
# copying node_modules alone would leave that symlink dangling in this
# stage, so /infra/kms-adapter needs to exist here at the identical path.
COPY --from=build /infra/kms-adapter /infra/kms-adapter
COPY --from=build /apps/api-gateway/node_modules ./node_modules
COPY --from=build /apps/api-gateway/dist ./dist
COPY --from=build /apps/api-gateway/package.json ./package.json
COPY apps/api-gateway/drizzle ./drizzle
COPY apps/api-gateway/kms-keys ./kms-keys

COPY apps/api-gateway/docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

EXPOSE 4000
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["node", "dist/index.js"]
