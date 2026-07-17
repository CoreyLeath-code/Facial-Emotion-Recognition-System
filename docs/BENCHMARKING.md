# Benchmark and Evaluation Guide

Run the deterministic input-boundary benchmark:

```bash
pytest tests/production/test_benchmark.py --benchmark-only \
  --benchmark-json=benchmarks/latest.json --no-cov
```

Record commit, Python/PyTorch versions, CPU/GPU, model checksum, batch size, input resolution, warm-up, iterations, min/mean/median/P95/P99/max latency, throughput, peak RSS, and GPU memory. Compare only like-for-like environments.

Model evaluation additionally requires a versioned held-out corpus and must report accuracy, macro/weighted precision, recall, F1, per-class support, confusion matrix, calibration, and subgroup limitations. MAP/MRR/NDCG and ROC-AUC are not primary metrics for this single-label seven-class classifier unless a specific ranked or one-vs-rest protocol is documented.
