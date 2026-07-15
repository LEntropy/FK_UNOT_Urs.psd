# delivery-gateway: standalone Rust (axum) binary, no other services'
# build artifacts needed (unlike protection-svc's rust-core, which is
# vendored into a Python image -- this one ships as its own image).
FROM rust:1-slim AS build
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim AS runtime
# reqwest's default TLS backend (rustls, see Cargo.toml) needs no OpenSSL
# system libs, but ca-certificates is still needed to validate any real
# HTTPS asset-service endpoint in a non-local deployment. wget is only for
# docker-compose's healthcheck (CMD wget ... /health) -- not otherwise used.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /app/target/release/delivery-gateway ./delivery-gateway
EXPOSE 4500
CMD ["./delivery-gateway"]
