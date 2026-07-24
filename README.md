# Facial Emotion Recognition System

[![CI](https://github.com/CoreyLeath-code/Facial-Emotion-Recognition-System/actions/workflows/ci.yml/badge.svg)](https://github.com/CoreyLeath-code/Facial-Emotion-Recognition-System/actions/workflows/ci.yml)
[![CodeQL](https://github.com/CoreyLeath-code/Facial-Emotion-Recognition-System/actions/workflows/codeql.yml/badge.svg)](https://github.com/CoreyLeath-code/Facial-Emotion-Recognition-System/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A PyTorch/FastAPI service for seven-class facial-expression classification on FER-style 48x48 grayscale inputs. The supported production boundary validates uploaded images, loads reviewed state dictionaries, and exposes separate liveness and readiness probes.

> Facial expressions do not reliably reveal internal emotional state. This project is for research and demonstration, not medical diagnosis or consequential decisions.


## Production Readiness Guide

> This section is the portfolio audit entry point for **Facial-Emotion-Recognition-System**. It describes an engineering promotion path; it is not a claim that the repository is already production-authorized.

[![CI](https://img.shields.io/github/actions/workflow/status/CoreyLeath-code/Facial-Emotion-Recognition-System/ci.yml?branch=main&label=CI)](https://github.com/CoreyLeath-code/Facial-Emotion-Recognition-System/actions) [![License](https://img.shields.io/github/license/CoreyLeath-code/Facial-Emotion-Recognition-System)](https://github.com/CoreyLeath-code/Facial-Emotion-Recognition-System/blob/main/LICENSE)

### Architecture flowchart

```mermaid
flowchart LR
    Client --> Gateway --> Services[API + workers] --> Events[(Event bus)] --> Store[(State)]
```

### Quickstart and local validation

The supported local path should be reproducible from a clean checkout. The inferred stack for this repository is **Python/platform services**.

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
pytest -q
```

If the project uses external services, model artifacts, cloud credentials, or private data, start them through documented local fixtures or mocks. Never place secrets or identifiable records in the repository.

### Research-style metrics and benchmarks

| Evidence | Required record |
|---|---|
| Correctness | Test command, commit SHA, runtime, and pass/fail result |
| Performance | Warm-up, sample count, concurrency, median, p95, p99, throughput, and memory |
| Data/model quality | Dataset version, split strategy, leakage controls, calibration, subgroup results, and uncertainty |
| Runtime | Image digest, health-check latency, resource limits, and rollback target |
| Security | Dependency, secret, SAST, container, and SBOM results |

A benchmark number belongs in a versioned artifact tied to a commit and hardware/runtime description. Engineering benchmarks must not be presented as clinical, financial, safety, or model-quality validation without the appropriate domain evidence.

### Extended Q&A

**What is production-ready for this repository?**  
A reproducible build, tested public contract, controlled configuration, observable runtime, documented security boundary, versioned artifacts, and a tested rollback path.

**What must remain explicit?**  
The intended use, excluded use, data/credential handling, model or algorithm limitations, and which metrics are measured versus aspirational.

**What should be completed next?**  
Use the linked production-readiness issue for this repository as the checklist. Resolve missing tests, deployment instructions, observability, supply-chain controls, and release evidence before attaching a production claim.


## Architecture

```mermaid
flowchart LR
  U["JPEG / PNG / WebP"] --> V["Bounded validation"]
  V --> P["48x48 preprocessing"]
  P --> T["PyTorch inference"]
  T --> J["Ranked emotion labels"]
  W["Read-only versioned weights"] --> T
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Run the API after placing reviewed weights at `artifacts/models/emotion_model.pt`:

```bash
pip install -e .
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

## Metrics dashboard

| Metric | Verified value |
|---|---:|
| Supported Python | 3.10-3.12 |
| Emotion classes | 7 |
| Input upload limit | 5 MiB default |
| Hardened boundary tests | 19 passing |
| Hardened boundary coverage | 100% branch coverage |
| Validation median latency | 92 microseconds (local) |
| Model accuracy | Pending reproducible evaluation |
| Inference latency/throughput | Pending versioned weights benchmark |
| Security findings | Published by CI |
| Docker image size | Published by CI/build system |

## Engineering controls

Pull requests run scoped formatting/linting, strict typing, tests and coverage, package/container builds, dependency and static security audits, secret scanning, license inventory, SBOM generation, CodeQL, and a reproducible benchmark. Optional profiles keep dashboard, training, vision, and LLM dependencies out of the minimal API installation.

## Documentation

- [Production audit](docs/AUDIT.md)
- [Deployment guide](docs/DEPLOYMENT.md)
- [Benchmark methodology](docs/BENCHMARKING.md)
- [Model card](MODEL_CARD.md)
- [Security and threat model](SECURITY.md)
- [Ethics](ethics.md)

See the audit before using legacy Streamlit, Kubernetes, Helm, Airflow, Snowflake, MLflow, RAG, or LLM prototypes.
