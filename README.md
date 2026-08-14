# Facial Expression Classification - Reproducible Research Scaffold

[![CI](https://github.com/CoreyLeath-code/Facial-Emotion-Recognition-System/actions/workflows/ci.yml/badge.svg)](https://github.com/CoreyLeath-code/Facial-Emotion-Recognition-System/actions/workflows/ci.yml)
[![CodeQL](https://github.com/CoreyLeath-code/Facial-Emotion-Recognition-System/actions/workflows/codeql.yml/badge.svg)](https://github.com/CoreyLeath-code/Facial-Emotion-Recognition-System/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10--3.12-3776AB)](pyproject.toml)

## Abstract

This repository contains a PyTorch seven-class facial-expression classifier for FER-style 48x48 grayscale images, a FastAPI inference boundary, and research-oriented reference implementations for stable softmax, cross entropy, confusion matrices, and macro metrics.

No reviewed checkpoint and immutable held-out dataset artifact are committed. Therefore, accuracy, macro F1, calibration, latency, and throughput are intentionally reported as TBD. This is a reproducible research scaffold and engineering demonstration, not a validated emotion-measurement system.

> Facial expressions do not reliably reveal a person's internal emotional state. Do not use this project for medical, employment, education, policing, surveillance, access-control, or other consequential decisions.

## Research questions

1. Does the PyTorch CNN improve on a majority-class baseline on the immutable FER2013 PrivateTest split?
2. How do normalization, CNN capacity, and seed affect macro F1 and per-class recall?
3. Which classes produce stable confusion patterns across repeated runs?
4. What accuracy-latency tradeoff appears under fixed weights, hardware, and batch settings?

## Formal problem statement

The supported model maps an image x to seven logits z = f_theta(x). Inference turns them into class probabilities with stable softmax and selects the largest probability. Cross-entropy is the training loss represented by the legacy TensorFlow script; its framework mismatch with the supported PyTorch inference path is an explicit limitation.

See [mathematical foundations](docs/MATHEMATICAL_FOUNDATIONS.md) for notation, equations, numerical-stability reasoning, and code mapping. See [complexity analysis](docs/COMPLEXITY_ANALYSIS.md) for convolution, dense-layer, softmax, and metric costs.

## Method and reference checks

The supported API path uses the PyTorch EmotionCNN implementation in src/src/modeling/model.py. It emits logits, applies softmax at inference time, and maps the seven indices to expression labels.

The separate src/research/classification_reference.py module is a dependency-free educational baseline. It does not replace optimized PyTorch operations. Its analytical tests verify:

- softmax normalization and invariance under a constant logit shift;
- cross entropy for a uniform two-class case;
- exact confusion-matrix entries;
- accuracy, macro precision, macro recall, and macro F1 for a hand-derived example;
- invalid-input handling.

## Evidence status

| Evidence | Status |
| --- | --- |
| API upload validation and readiness behavior | Exercised by tests/production in CI |
| Mathematical softmax/loss/metric reference | Analytically tested |
| Model quality on held-out data | TBD: no reviewed checkpoint/corpus artifact |
| Repeated-seed statistics and confidence intervals | TBD |
| Calibration, subgroup analysis, and robustness | TBD |
| API latency and throughput | TBD: benchmark protocol only |

## Reproducibility

Create the supported API/test environment:

~~~bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest
~~~

The research evidence contract is in [experiments/README.md](experiments/README.md). A future result must record the commit, checkpoint checksum, dataset and split identifiers, preprocessing, seed, versions, device, configuration, and individual-run metrics. Do not replace TBD values without that machine-readable artifact.

## Engineering architecture

~~~mermaid
flowchart LR
  I["Image upload"] --> V["Bounded validation"]
  V --> P["48x48 preprocessing"]
  P --> M["PyTorch logit model"]
  M --> R["Ranked expression labels"]
  W["Reviewed read-only weights"] --> M
~~~

The reviewed engineering boundary is the FastAPI inference service: upload validation, liveness/readiness endpoints, typed configuration, CI, package/container build, dependency audit, secret scanning, license inventory, SBOM generation, and CodeQL. Legacy TensorFlow training, Streamlit, RAG/LLM, Snowflake, Airflow, and deployment prototypes are not an integrated supported system.

## Limitations and threats to validity

- Training code and supported inference use different frameworks and must be reconciled before an end-to-end result is credible.
- A random re-split of the full CSV is not a final leakage-safe FER2013 evaluation protocol; preserve Usage partitions.
- FER-style labels are ambiguous, culturally variable, and not ground truth for a person's internal state.
- No committed artifact supports accuracy, calibration, demographic, robustness, or systems-performance claims.
- The repository has duplicate source layouts that increase maintenance and compatibility risk.

## Documentation

- [Academic audit](docs/ACADEMIC_AUDIT.md)
- [Mathematical foundations](docs/MATHEMATICAL_FOUNDATIONS.md)
- [Complexity analysis](docs/COMPLEXITY_ANALYSIS.md)
- [Research roadmap](docs/RESEARCH_ROADMAP.md)
- [Benchmark methodology](docs/BENCHMARKING.md)
- [Evaluation artifact contract](experiments/README.md)
- [Deployment audit](docs/AUDIT.md)
- [Model card](MODEL_CARD.md)

## License

MIT. See [LICENSE](LICENSE).
