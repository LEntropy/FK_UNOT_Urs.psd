# web: static Vite build served by server.mjs's tiny Express server.
# VITE_API_GATEWAY_URL is baked in at *build* time (Vite convention -- it's
# client-side JS running in the user's browser, not container-to-container
# traffic) -- defaults to localhost:4000 to match docker-compose's port
# mapping for api-gateway, since the browser runs on the host, not inside
# the compose network.
FROM node:22-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.json vite.config.ts index.html ./
COPY src ./src
ARG VITE_API_GATEWAY_URL=http://localhost:4000
ENV VITE_API_GATEWAY_URL=${VITE_API_GATEWAY_URL}
RUN npm run build

FROM node:22-slim AS runtime
WORKDIR /app
ENV NODE_ENV=production
COPY package.json package-lock.json ./
RUN npm ci --omit=dev
COPY server.mjs ./
COPY --from=build /app/dist ./dist

EXPOSE 5173
CMD ["node", "server.mjs"]
