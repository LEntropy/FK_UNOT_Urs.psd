# blockchain-svc: Node/TypeScript, ethers.js, no native deps -- but does
# have a `file:` dependency on ../../infra/kms-adapter (the KMS-backed
# relayer key envelope-encryption path, INTEGRATION.md's "Relayer key"
# section) since task #35. Build context must be the repo root, not
# apps/blockchain-svc alone, so that dependency actually resolves inside
# the image (mirrors the real directory structure: /infra, /apps/
# blockchain-svc -- same reasoning as apps/asset-service/Dockerfile and
# apps/api-gateway/Dockerfile). A context scoped to apps/blockchain-svc
# alone doesn't just fail to find the package -- npm ci's Arborist
# crashes trying to load the missing link node from the lockfile's
# virtual tree and reports a misleading "no existing package-lock.json"
# error instead of the real cause, found the hard way running this
# Dockerfile with the old (wrong) context.
FROM node:22-slim AS build

WORKDIR /infra/kms-adapter
COPY infra/kms-adapter/package.json infra/kms-adapter/package-lock.json ./
RUN npm ci
COPY infra/kms-adapter/tsconfig.json ./
COPY infra/kms-adapter/src ./src
RUN npm run build

WORKDIR /apps/blockchain-svc
COPY apps/blockchain-svc/package.json apps/blockchain-svc/package-lock.json ./
RUN npm ci
COPY apps/blockchain-svc/tsconfig.json ./
COPY apps/blockchain-svc/src ./src
RUN npm run build

FROM node:22-slim AS runtime
WORKDIR /apps/blockchain-svc
ENV NODE_ENV=production
# node_modules/@dontai/kms-adapter is a symlink to /infra/kms-adapter --
# copying node_modules alone would leave it dangling in this stage.
COPY --from=build /infra/kms-adapter /infra/kms-adapter
COPY --from=build /apps/blockchain-svc/node_modules ./node_modules
COPY --from=build /apps/blockchain-svc/dist ./dist
COPY --from=build /apps/blockchain-svc/package.json ./package.json
COPY apps/blockchain-svc/docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

EXPOSE 3001
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["node", "dist/index.js"]
