# Security Policy and Threat Model

Report vulnerabilities through GitHub private vulnerability reporting; do not open a public issue. Include reproduction steps, affected versions, impact, and mitigation ideas.

The API treats images, filenames, content types, model files, environment variables, and upstream LLM responses as untrusted. Principal threats are decompression bombs, oversized uploads, malicious model serialization, dependency compromise, denial of service, privacy leakage, prompt injection through optional LLM features, and unauthorized biometric processing.

Controls include bounded image validation, weights-only model loading, read-only artifacts/filesystem, non-root containers, dropped Linux capabilities, restricted CORS, dependency auditing, secret scanning, CodeQL, and SBOM generation. Before public deployment add authentication, authorization, TLS, rate limiting, request timeouts, retention controls, audit logs, and an incident-response owner.

Raw images must not be logged. Obtain informed consent and define deletion periods. Facial-expression classification is not a reliable measurement of internal mental state and must not drive consequential decisions.
