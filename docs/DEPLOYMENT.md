# Deployment Guide

The supported artifact is the FastAPI container.

```bash
docker build -t facial-emotion-recognition:local .
docker run --rm -p 8000:8000 \
  -v "$PWD/artifacts/models:/models:ro" \
  -e MODEL_PATH=/models/emotion_model.pt \
  facial-emotion-recognition:local
```

`GET /health/live` verifies the process. `GET /health/ready` returns success only when model weights loaded. Never route production traffic using liveness alone.

Production requires an ingress body limit no greater than `MAX_UPLOAD_BYTES`, TLS, authentication, per-client rate limiting, timeouts, restricted egress, immutable image/model digests, CPU/memory limits, centralized redacted logs, latency/error/saturation alerts, and a retained previous release for rollback.

The legacy Kubernetes and Helm prototypes are not approved for deployment until reconciled with this API-only contract.
