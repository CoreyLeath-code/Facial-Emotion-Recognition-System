# Reproducible evaluation artifact contract

The repository does not contain a reviewed model checkpoint and held-out FER2013 corpus, so it must not publish accuracy, F1, calibration, or subgroup values. This directory defines evidence required for a future evaluation.

A machine-readable result must include commit SHA, UTC timestamp, model artifact checksum, architecture, dataset source/version/checksum, split identifier, preprocessing contract, class map, seed, Python/PyTorch versions, device, batch size, and per-run accuracy, macro precision/recall/F1, confusion matrix, and calibration method. Keep every seed as an individual record; aggregate mean and sample standard deviation only after repeated runs.

FER2013 Usage should be preserved as the train, validation, and test protocol rather than resampling the full CSV for a final score. The legacy TensorFlow script resamples its full input and is not evidence of leakage-safe final evaluation.

Example command once reviewed data and compatible PyTorch weights exist:

~~~bash
PYTHONPATH=. python src/evaluate.py --model cnn
~~~

Before reporting values, update evaluation to accept the immutable test split and write a JSON artifact here. Do not replace TBD values in the README without that artifact.
