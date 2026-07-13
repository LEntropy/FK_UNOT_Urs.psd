# detection-svc reuses two things from protection-svc rather than
# duplicating them: rust-core's `detect` binary, and ml-engine's
# perceptual_hash.py module. Build context for this Dockerfile is the repo
# root (see docker-compose.yml) so both are reachable.
#
# See apps/protection-svc/Dockerfile's header comment: a cross-compile
# optimization here was tried and reverted after it caused build failures
# without ever producing a working image. Plain QEMU-emulated build below.

FROM rust:1-slim AS rust-build
WORKDIR /rust-core
COPY apps/protection-svc/rust-core/Cargo.toml apps/protection-svc/rust-core/Cargo.lock ./
COPY apps/protection-svc/rust-core/src ./src
RUN cargo build --release

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
COPY --from=rust-build /rust-core/target/release/rust-core ../protection-svc/rust-core/target/release/rust-core

EXPOSE 8003
CMD ["python", "server.py"]
