# delivery-gateway: standalone Rust (axum) binary, no other services'
# build artifacts needed (unlike protection-svc's rust-core, which is
# vendored into a Python image -- this one ships as its own image).
# Pinned to bookworm explicitly, matching the runtime stage's debian:
# bookworm-slim -- plain `rust:1-slim` now tracks Debian 13 (trixie,
# glibc 2.38+), which produced a binary the bookworm (glibc 2.36) runtime
# stage couldn't run at all ("GLIBC_2.38' not found"), found live
# bringing this stack up locally.
FROM rust:1-slim-bookworm AS build
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
