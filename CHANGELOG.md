# Changelog

All notable release-level changes to Facial Emotion Recognition System are documented here.

## [Unreleased]

## [0.2.0] - 2026-08-21

### Added
- Installable `facial-emotion-recognition-system` Python package with FastAPI inference-service dependencies.
- CI quality gates for Ruff formatting/linting, strict mypy checks, pytest, and a 90% coverage floor on the production-scoped modules.
- Package and Docker-image build verification.
- Security checks including Bandit, dependency auditing, Gitleaks, license inventory, and SPDX SBOM generation.
- Reproducible pytest-benchmark execution with JSON artifacts.
- Tagged release automation that builds Python distributions, publishes the application image to GHCR, and creates a GitHub Release.

### Scope
- Version `0.2.0` reflects the package metadata in `pyproject.toml`.
- This release packages the repository's implemented inference service and verification surface; it does not claim demographic fairness, clinical validity, affective-state certainty, or production accuracy beyond versioned evidence in the repository.
