# Academic audit

## Scope and evidence boundary

This is a computer-vision repository for seven-way facial-expression classification on FER-style 48x48 grayscale inputs. The supported inference path is the PyTorch EmotionCNN and EmotionModel implementation under src/src/modeling/model.py; it emits seven logits and applies softmax only for prediction.

The repository also contains legacy TensorFlow/Keras training and evaluation scripts, plus optional RAG/LLM, dashboard, and infrastructure prototypes. They should not be treated as one validated end-to-end system. Facial expression labels do not reliably reveal internal emotional state, and this project is unsuitable for medical, employment, education, policing, surveillance, access-control, or other consequential decisions.

## Admissions-oriented assessment

| Dimension | Current evidence | Assessment |
| --- | --- | --- |
| Algorithmic depth | Direct CNN layers, logits, preprocessing, and a pure-Python reference for softmax, loss, and metrics | Partial: optimized convolution/backpropagation remains delegated to frameworks |
| Mathematical rigor | Logit semantics and evaluation equations map to source files | Improved, but no verified checkpoint lineage |
| Scientific evaluation | Evaluation code and protocol exist; no versioned held-out result is committed | Weak: quality values remain TBD |
| Reproducibility | Package, CI, deterministic reference tests, and artifact fields | Partial: no reviewed dataset/checkpoint artifact |
| Systems engineering | FastAPI validation, readiness, CI, package/container checks, supply-chain scans | Stronger than research evidence; only the API boundary is reviewed |

## Claims audit

| Claim area | Classification | Rationale |
| --- | --- | --- |
| Seven output classes and 48x48 CNN input | DERIVED | Defined in model code and label map |
| Input boundary behavior | MEASURED | Tests in tests/production run in CI |
| Accuracy, macro F1, calibration, latency, throughput | ASPIRATIONAL | No versioned checkpoint, corpus, or machine-readable result |
| Whole-system production readiness | UNSUPPORTED | API boundary is hardened; training, RAG/LLM, and infrastructure remain prototypes |

## Material weaknesses

1. train.py uses TensorFlow while the supported API uses PyTorch, so a TensorFlow artifact is not API-compatible.
2. Legacy training uses a random split over its input CSV; the FER2013 Usage partitions should be retained for final evaluation.
3. No model checksum, dataset checksum, exact split manifest, or measured held-out result is committed.
4. Per-class errors, calibration, demographic coverage, and robustness to pose, lighting, and occlusion are unmeasured.
5. Duplicate module layouts obscure the supported path.

## Research questions

1. Does the PyTorch CNN exceed a majority-class baseline on the immutable PrivateTest split in macro F1 and per-class recall?
2. How does normalization choice affect performance when architecture and split are fixed?
3. Which classes dominate the confusion matrix across independent seeds?
4. How sensitive are accuracy and macro F1 to seed, batch size, and model capacity?
5. At what batch size/device configuration does throughput improve without unacceptable per-request latency?
