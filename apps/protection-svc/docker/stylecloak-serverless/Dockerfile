# syntax=docker/dockerfile:1
#
# Serverless variant of docker/stylecloak/Dockerfile -- same pattern as
# docker/strongprotect-serverless/Dockerfile (reuse the base image,
# add the runpod SDK, swap CMD to the job handler).
#
# Build (from apps/protection-svc/docker/stylecloak-serverless/):
#   docker build -t lentropy/dontai-stylecloak-serverless:latest .
#   docker push lentropy/dontai-stylecloak-serverless:latest

FROM lentropy/dontai-stylecloak:latest

RUN pip install --no-cache-dir --break-system-packages runpod

COPY handler.py /workspace/handler.py

CMD ["python3", "/workspace/handler.py"]
