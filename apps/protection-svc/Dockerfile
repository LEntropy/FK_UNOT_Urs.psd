# protection-svc: FastAPI (Python) orchestrating ml-engine (PyTorch cloak)
# and rust-core (watermark/variants CLI, built as a release binary and
# shelled out to -- see orchestrate.py's RUST_CORE_BIN candidates).
#
# Multi-arch (amd64+arm64) builds via buildx default to QEMU-emulating the
# ENTIRE Dockerfile for the non-native target -- brutal for `cargo build`
# (CPU-bound compilation under an emulated CPU is often 5-10x slower) and
# still slow for a big pip install. Both heavy stages below instead run
# natively via `--platform=$BUILDPLATFORM` and cross-compile/cross-fetch
# for the actual target ($TARGETARCH), so nothing CPU-heavy runs under
# emulation -- only the final, cheap COPY-only assembly stage is emulated.

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
COPY rust-core/Cargo.toml rust-core/Cargo.lock ./
COPY rust-core/src ./src
RUN cargo build --release --target "$(cat /rust_target)" \
    && cp "target/$(cat /rust_target)/release/rust-core" /rust-core-bin

# Fetch prebuilt wheels for the TARGET arch natively too (pip download just
# resolves+downloads matching wheel files -- no code from them executes, so
# this doesn't need emulation even though the wheels themselves are arm64).
FROM --platform=$BUILDPLATFORM python:3.11-slim AS wheel-fetch
ARG TARGETARCH
WORKDIR /wheels
RUN case "$TARGETARCH" in \
      amd64) echo manylinux2014_x86_64 > /py_platform ;; \
      arm64) echo manylinux2014_aarch64 > /py_platform ;; \
      *) echo "unsupported TARGETARCH $TARGETARCH" >&2; exit 1 ;; \
    esac
COPY ml-engine/requirements.txt ./requirements.txt
# torch/torchvision are downloaded separately from the CPU-only index below
# and stripped from this second requirements-based download -- otherwise
# pip would re-resolve them here against plain PyPI, which is exactly how
# this project ended up with an unwanted CUDA build once before (see
# ml-engine/README.md's install notes).
RUN grep -vE "^(torch|torchvision)" requirements.txt > requirements-notorch.txt
RUN pip download --no-deps --only-binary=:all: --python-version 3.11 \
      --platform "$(cat /py_platform)" \
      --index-url https://download.pytorch.org/whl/cpu \
      -d . torch torchvision \
    && pip download --only-binary=:all: --python-version 3.11 \
      --platform "$(cat /py_platform)" \
      -d . -r requirements-notorch.txt

FROM python:3.11-slim AS runtime
WORKDIR /app

COPY --from=wheel-fetch /wheels /wheels
COPY ml-engine/requirements.txt ./ml-engine/requirements.txt
RUN pip install --no-cache-dir --no-index --find-links=/wheels torch torchvision \
    && pip install --no-cache-dir --no-index --find-links=/wheels -r ml-engine/requirements.txt \
    && rm -rf /wheels

COPY orchestrate.py server.py remote_gpu.py ./
COPY ml-engine/src ./ml-engine/src

# orchestrate.py's default RUST_CORE_BIN candidates already look for
# rust-core/target/release/rust-core (no extension, Linux naming).
COPY --from=rust-build /rust-core-bin ./rust-core/target/release/rust-core

EXPOSE 8000
CMD ["python", "server.py"]
