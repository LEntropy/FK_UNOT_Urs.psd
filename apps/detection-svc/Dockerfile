# detection-svc reuses two things from protection-svc rather than
# duplicating them: rust-core's `detect` binary, and ml-engine's
# perceptual_hash.py module. Build context for this Dockerfile is the repo
# root (see docker-compose.yml) so both are reachable.
#
# rust-build runs natively via --platform=$BUILDPLATFORM and cross-compiles
# to $TARGETARCH instead of letting buildx QEMU-emulate the whole stage --
# `cargo build` under emulation is the single slowest part of a multi-arch
# build here (see apps/protection-svc/Dockerfile's header comment for the
# full rationale).

FROM --platform=$BUILDPLATFORM rust:1-slim AS rust-build
ARG TARGETARCH
WORKDIR /rust-core
RUN case "$TARGETARCH" in \
      amd64) echo x86_64-unknown-linux-gnu > /rust_target ;; \
      arm64) echo aarch64-unknown-linux-gnu > /rust_target ;; \
      *) echo "unsupported TARGETARCH $TARGETARCH" >&2; exit 1 ;; \
    esac
RUN rustup target add "$(cat /rust_target)"
RUN if [ "$TARGETARCH" = "arm64" ]; then \
      apt-get update && apt-get install -y --no-install-recommends gcc-aarch64-linux-gnu && rm -rf /var/lib/apt/lists/* ; \
    fi
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
COPY apps/protection-svc/rust-core/Cargo.toml apps/protection-svc/rust-core/Cargo.lock ./
COPY apps/protection-svc/rust-core/src ./src
RUN cargo build --release --target "$(cat /rust_target)" \
    && cp "target/$(cat /rust_target)/release/rust-core" /rust-core-bin

FROM python:3.11-slim AS runtime
WORKDIR /app

COPY apps/detection-svc/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m playwright install --with-deps chromium

COPY apps/detection-svc/server.py ./
COPY apps/detection-svc/src ./src

# phash_match.py imports perceptual_hash.py via a relative sys.path lookup
# at apps/protection-svc/ml-engine/src (see src/phash_match.py) -- reuse the
# same relative layout inside the image instead of hardcoding a different
# path there.
COPY apps/protection-svc/ml-engine/src/perceptual_hash.py ../protection-svc/ml-engine/src/perceptual_hash.py

# rust_watermark.py's default candidates also expect this relative layout
# (../protection-svc/rust-core/target/release/rust-core from src/).
COPY --from=rust-build /rust-core-bin ../protection-svc/rust-core/target/release/rust-core

EXPOSE 8003
CMD ["python", "server.py"]
