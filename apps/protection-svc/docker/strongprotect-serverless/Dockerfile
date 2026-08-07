# syntax=docker/dockerfile:1
#
# Serverless variant of docker/strongprotect/Dockerfile, built purely to
# empirically compare RunPod Serverless against the Pods-based approach
# (PHASE4_SCOPING.md SS6, 2026-08-07 update) -- cost, cold-start time, and
# operational reliability (Pods needs us to hand-roll create/poll/SSH/
# delete lifecycle management; Serverless's queue+autoscaler does that).
#
# Reuses the already-built and pushed lentropy/dontai-strongprotect:latest
# as a base (checkpoints + attack scripts already baked in, ~36GB) --
# this layer just adds the runpod SDK and swaps the CMD from sshd to the
# Serverless job handler. Pods image is untouched/still the production path
# for strong_protection; this is a side-by-side comparison artifact.
#
# Build (from apps/protection-svc/docker/strongprotect-serverless/):
#   docker build -t lentropy/dontai-strongprotect-serverless:latest .
#   docker push lentropy/dontai-strongprotect-serverless:latest

FROM lentropy/dontai-strongprotect:latest

RUN pip install --no-cache-dir --break-system-packages runpod

COPY handler.py /workspace/handler.py

CMD ["python3", "/workspace/handler.py"]
