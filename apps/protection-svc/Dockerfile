# protection-svc: FastAPI (Python) orchestrating ml-engine (PyTorch cloak)
# and rust-core (watermark/variants CLI, built as a release binary and
# shelled out to -- see orchestrate.py's RUST_CORE_BIN candidates).

FROM rust:1-slim AS rust-build
WORKDIR /rust-core
COPY rust-core/Cargo.toml rust-core/Cargo.lock ./
COPY rust-core/src ./src
RUN cargo build --release

FROM python:3.11-slim AS runtime
WORKDIR /app

# CPU-only torch build (documented path in ml-engine/README.md) -- this
# image has no GPU, and USE_REMOTE_GPU is off by default in docker-compose
# (no GPU PC reachable from a generic dev host), so cloak runs on CPU here:
# slower, still correct.
COPY ml-engine/requirements.txt ./ml-engine/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir -r ml-engine/requirements.txt

COPY orchestrate.py server.py remote_gpu.py ./
COPY ml-engine/src ./ml-engine/src

# orchestrate.py's default RUST_CORE_BIN candidates already look for
# rust-core/target/release/rust-core (no extension, Linux naming).
COPY --from=rust-build /rust-core/target/release/rust-core ./rust-core/target/release/rust-core

EXPOSE 8000
CMD ["python", "server.py"]
