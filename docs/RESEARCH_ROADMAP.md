# Research roadmap

## Phase 1 - compatibility and lineage

Choose one supported training framework, make preprocessing identical between training and inference, and publish a manifest linking commit, checkpoint checksum, class map, data source/version/checksum, and split identifiers.

## Phase 2 - leakage-safe baseline

Preserve FER2013 published usage partitions. Evaluate a majority-class baseline and the CNN on the immutable held-out partition; write one JSON record per seed under artifacts/experiments.

## Phase 3 - controlled comparison

Hold split and preprocessing fixed while varying one factor at a time: normalization, CNN capacity, augmentation, and optimizer settings. Report macro F1, per-class recall, confusion matrices, and runtime separately.

## Phase 4 - uncertainty and failure analysis

Use repeated seeds, report mean/sample standard deviation, inspect calibration, and disclose class imbalance, label ambiguity, pose, lighting, occlusion, and demographic-coverage limitations.

## Phase 5 - systems evidence

Measure API latency and throughput using fixed images, versioned weights, explicit concurrency/batch settings, and documented device. Keep systems benchmarks separate from model-quality evaluation.
