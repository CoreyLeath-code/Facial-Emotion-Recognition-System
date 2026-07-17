# Benchmark Report

The local result below covers the hardened image-validation boundary, not model inference. CI publishes `latest.json` for each commit. Model inference results remain pending a versioned weights artifact and fixed runner profile; undocumented hardware claims are not treated as evidence.

| Metric | Value |
|---|---:|
| Benchmark date | 2026-07-17 |
| Input | 224x224 JPEG |
| Minimum latency | 84.2 microseconds |
| Mean latency | 119.5 microseconds |
| Median latency | 92.0 microseconds |
| Maximum latency | 323.4 microseconds |
| Throughput | 8,366 validations/sec |
| Peak RAM / GPU RAM | Pending |
| Model checksum | Not supplied |
