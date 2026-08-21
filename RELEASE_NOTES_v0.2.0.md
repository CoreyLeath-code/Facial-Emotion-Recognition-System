# Facial Emotion Recognition System v0.2.0

This is the first formal portfolio release prepared from the repository's existing `0.2.0` package metadata and CI surface.

## Verified release surface

The repository declares package version `0.2.0`, Python `>=3.10,<3.13`, and a FastAPI-oriented dependency set in `pyproject.toml`.

The existing CI verifies:

- Ruff formatting and linting;
- strict mypy checks on production-scoped modules;
- pytest with a 90% coverage floor;
- Python package builds;
- Docker image builds;
- Bandit, pip-audit, Gitleaks, license inventory, and SPDX SBOM generation;
- a pytest-benchmark run with a JSON artifact.

## Publishing contract

A successful `v0.2.0` tag will:

1. validate that the tag matches `pyproject.toml` and the changelog;
2. run the production-scoped test suite and package/container build checks;
3. build wheel and source distributions;
4. create the GitHub Release with those distributions attached; and
5. publish the validated container to `ghcr.io/coreyleath-code/facial-emotion-recognition-system` with semantic-version and `latest` tags.

## Responsible-use scope

This release does not claim demographic fairness, clinical or psychological validity, affective-state certainty, or production accuracy beyond evidence explicitly stored and reproducible in the repository. Facial-expression classification should not be treated as a reliable proxy for a person's internal emotional state.
