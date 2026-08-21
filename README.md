# Facial Expression Classification — Reproducible Research Scaffold

<p align="center">
  <a href="https://github.com/CoreyLeath-code/Facial-Emotion-Recognition-System/actions/workflows/ci.yml">
    <img alt="CI on main" src="https://github.com/CoreyLeath-code/Facial-Emotion-Recognition-System/actions/workflows/ci.yml/badge.svg?branch=main">
  </a>
  <a href="https://github.com/CoreyLeath-code/Facial-Emotion-Recognition-System/actions/workflows/codeql.yml">
    <img alt="CodeQL on main" src="https://github.com/CoreyLeath-code/Facial-Emotion-Recognition-System/actions/workflows/codeql.yml/badge.svg?branch=main">
  </a>
  <a href="https://github.com/CoreyLeath-code/Facial-Emotion-Recognition-System/releases/latest">
    <img alt="Latest release" src="https://img.shields.io/github/v/release/CoreyLeath-code/Facial-Emotion-Recognition-System?display_name=tag&sort=semver">
  </a>
  <a href="https://github.com/CoreyLeath-code/Facial-Emotion-Recognition-System/pkgs/container/facial-emotion-recognition-system">
    <img alt="GHCR container" src="https://img.shields.io/badge/GHCR-container-2496ED?logo=docker&logoColor=white">
  </a>
  <a href="LICENSE">
    <img alt="License MIT" src="https://img.shields.io/badge/license-MIT-blue.svg">
  </a>
  <a href="pyproject.toml">
    <img alt="Python 3.10 to 3.12" src="https://img.shields.io/badge/python-3.10--3.12-3776AB?logo=python&logoColor=white">
  </a>
</p>

## Abstract

This repository contains a PyTorch seven-class facial-expression classifier for FER-style 48x48 grayscale images, a FastAPI inference boundary, and research-oriented reference implementations for stable softmax, cross entropy, confusion matrices, and macro metrics.

No reviewed checkpoint and immutable held-out dataset artifact are committed. Therefore, accuracy, macro F1, calibration, latency, and throughput are intentionally reported as TBD. This is a reproducible research scaffold and engineering demonstration, not a validated emotion-measurement system.

> Facial expressions do not reliably reveal a person's internal emotional state. Do not use this project for medical, employment, education, policing, surveillance, access-control, or other consequential decisions.

## Release and package

The current tagged release is **v0.2.0**. Release assets and notes are available from [GitHub Releases](https://github.com/CoreyLeath-code/Facial-Emotion-Recognition-System/releases/tag/v0.2.0). The container publishing workflow targets the GitHub Container Registry package at `ghcr.io/coreyleath-code/facial-emotion-recognition-system`.

~~~bash
docker pull ghcr.io/coreyleath-code/facial-emotion-recognition-system:0.2.0
~~~

The container package is an engineering distribution artifact; it does not change the model-evidence limitations documented below.

## Recent enhancements

- **Versioned v0.2.0 distribution:** the release workflow validates the semantic tag, package version, changelog, formatting, static types, tests, Python distributions, and container build before publishing release assets.
- **GHCR publishing and recovery:** tagged releases publish version, minor-version, and `latest` container tags. A manual workflow-dispatch path can validate an existing semantic tag and republish its container image if a tag event was missed; it does not create a new release.
- **Auditable CI evidence:** CI produces coverage, security-inventory (license and SPDX SBOM), and benchmark JSON artifacts. These artifacts document engineering checks; they are not model-quality or human-emotion validation evidence.
- **Honest research boundary:** the README, model card, benchmark contract, mathematical foundations, and complexity analysis explicitly distinguish verified software behavior from still-TBD model performance claims.

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

## Research benchmark and metric plan

All results below remain **TBD** until a reviewed checkpoint, immutable held-out corpus, and machine-readable run artifact exist. The protocol is designed to make a future result falsifiable and reproducible rather than to imply current model quality.

| Evaluation area | Required metrics | Current result | Evidence required before reporting |
| --- | --- | --- | --- |
| Single-label classification | Accuracy; macro and weighted precision, recall, F1; per-class support and recall; confusion matrix | TBD | Immutable test-split identifier/checksum, model checksum, preprocessing and class map |
| Uncertainty and calibration | Negative log likelihood, Brier score, expected calibration error, reliability diagram | TBD | Fixed calibration procedure and held-out evaluation predictions |
| Robustness and limitations | Defined perturbation or capture-condition protocol; per-class outcomes; failure analysis | TBD | Versioned protocol, input provenance, and raw prediction records |
| Inference performance | Warm-up, iterations, batch size, min/mean/median/P95/P99 latency, throughput, peak RSS, GPU memory | TBD | Commit, runtime versions, device, model checksum, and benchmark JSON |
| Repeated-run variability | Individual seed records, mean, sample standard deviation, confidence interval method | TBD | Fixed seeds and one artifact per run |

The repository’s existing deterministic input-boundary benchmark can be run with:

~~~bash
pytest tests/production/test_benchmark.py --benchmark-only \
  --benchmark-json=benchmarks/latest.json --no-cov
~~~

It is an engineering measurement only. It must record the environment described in [the benchmark guide](docs/BENCHMARKING.md), and it must not be presented as classification accuracy, emotion validity, or a deployment service-level objective.

### Reporting contract

A publishable evaluation must preserve FER2013 Usage partitions rather than re-splitting the full CSV, and must provide the commit SHA, UTC timestamp, model and dataset checksums, preprocessing, class map, seeds, Python/PyTorch versions, device, batch size, individual-run metrics, and confusion matrix. The [evaluation artifact contract](experiments/README.md) defines the required record. Numerical claims belong in a versioned artifact first and in this README only after review.

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

## Questions and answers

### Does this system measure a person’s internal emotional state?

No. It classifies image patterns into the repository’s seven expression-label categories. Expression labels are ambiguous and context-dependent; they are not reliable evidence of internal state.

### Why report macro F1 in addition to accuracy?

Accuracy can be dominated by more frequent classes. Macro F1 averages the per-class F1 values, giving each class equal weight and making uneven performance more visible. It still does not establish calibration, fairness, or real-world validity.

### Why are the model-quality metrics marked TBD?

The repository does not include a reviewed checkpoint plus an immutable held-out corpus artifact. Publishing a numerical score without those inputs, provenance, and per-run evidence would not be reproducible or credible.

### What would make a future benchmark credible?

Use the preserved FER2013 Usage partitions; freeze the checkpoint, dataset, preprocessing, and class map; evaluate multiple fixed seeds where training is involved; write raw predictions and a machine-readable result; and report the full metric set and limitations, not only a headline score.

### What does the API benchmark measure?

It measures only the documented input-boundary/inference engineering path under its recorded environment. Latency or throughput values cannot be compared across machines, batch sizes, model weights, or runtime versions without matching conditions.

### Is the FastAPI service ready for consequential deployment?

No. It provides an engineering inference boundary with validation and readiness behavior. The repository explicitly excludes medical, employment, education, policing, surveillance, access-control, and other consequential uses.

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
