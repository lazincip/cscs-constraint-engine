# Changelog

All notable changes to `cscs-constraint-engine` are documented in this file.

## [2026.10.1] - 2026-10-28 (Round 1 & Round 2 Audit Polish)

### Documentation (R3, pre-release — no code changes)
- Landing-page `README.md` (pitch, quickstart, Mermaid pipeline diagram, headline numbers) with depth relocated to `docs/ARCHITECTURE.md`, `docs/BENCHMARKS.md` (committed Track A baseline + protocol), and `docs/LIMITATIONS.md`. All verified claims and measured values carried over verbatim; code, tests, license, and metadata byte-identical to the R2 freeze candidate.

### Fixed
- **F1 (Long-Session Latency Lock — Verified $\sim 130\times$ Speedup at Turn 300):**
  - Cached 512-D prompt embeddings (`_embed_cache`, `maxlen=1024`) and 32-bit token digests (`_token_hash_cache`, `maxlen=8192`) in `SpectralManifold`, and cached vector/compressed-reference registries (`max_registry=64`, `zlib` level 1) in `MDLCompressionRouter`.
  - Replaced full $W \times 512$ SVD refit on every turn with exact Dual-Gram SVD (`G = X_c @ X_c.T` over sliding window `_svd_window = 20`, refit every turn for $N \le 10$ and novelty-gated at $\text{residual} > 0.30$ every 10 turns thereafter).
  - **Independent R2 Verification (`Linux VM, Python 3.12.3, numpy 2.5.4`):** Probe A (300 turns) median `1.071 ms` | mean `1.148 ms` | P95 `1.507 ms` (Turn 300 = `1.074 ms`, down from `~139 ms` in `v2026.10.0`). Probe B (300 turns) median `0.887 ms` | P95 `1.369 ms` | max `1.768 ms`.
- **F2 (Cross-Environment SVD Sign Canonicalization):**
  - Added `_canonicalize_signs(Vt)` in `SpectralManifold._fit_subspace()`, orienting each right singular vector $V_k[i]$ so $\arg\max_j |V_k[i, j]|$ is strictly positive. Verified cross-environment demo friction agreement to $\le 0.005$ on 3 of 4 turns (`0.448 / 0.669 / 0.634 / 0.726` vs. `0.448 / 0.674 / 0.634 / 0.726`).
- **F3 (Normalized Centered Reconstruction Residual):**
  - Normalized `reconstruction_error` by the centered norm $\|v_c\|_2 + \varepsilon$ instead of uncentered $\|v\|_2$, guaranteeing `reconstruction_error` $\in [0.0, 1.0]$ identically (Track A battery max residual verified at `0.9949 <= 1.0`).
- **F5 (Canonical Apache-2.0 License & NOTICE):**
  - Replaced truncated `LICENSE` with the full canonical Apache License 2.0 text (`11,357 bytes`, all 9 sections + appendix) for GitHub license auto-detection, and moved the Section 4(d) attribution notice to `NOTICE` (`520 bytes`).
- **F6 (T3LB Seed Precision Specification):**
  - Clarified docstrings, `README.md`, and unit tests to specify that `T3LBStateCodec` roundtrip serialization (`93 bytes`) is lossless at 2-decimal fixed precision.

### Changed (Backward-Compatible API & Semantic Knobs Record — C4)
- **Public API Signatures (Backward-Compatible):**
  - Added optional keyword argument `max_memory: int = 32` to `BanachContractionMirror.__init__()`.
  - Added optional keyword argument `max_registry: int = 64` to `MDLCompressionRouter.__init__()`.
  - Changed default `max_iter` from `10` to `8` in `BanachContractionMirror.reflect(prompt, max_iter: int = 8, tol: float = 1e-4)`.
- **Internal Semantic Knobs Adjusted Under F1:**
  - `SpectralManifold`: `history` `maxlen` `500 -> 64`, `_svd_window = 20`, novelty-gated refits after $N = 10$.
  - `RBFConstraintScout`: `nodes` `maxlen` `300 -> 64`, `ood_penalty` offset `0.35 -> 0.30`, `alpha` cutoff `rec_error >= 0.95`.
  - `BanachContractionMirror`: `memory_vectors` `maxlen` `50 -> 32`.
  - `MDLCompressionRouter`: `max_registry` `64`, `zlib` compression `level` `9 -> 1`.
  - **Net Behavioral Effect:** Perfect 4-decimal verdict-layer parity (`H1: 0.865`, `H2: 0.157`, `H3: 0.764`, demo Turns 1–4 identical to `v2026.10.0`), $130\times$ latency improvement at turn 300, and $0.019$ narrower held-out cold-start margin on Track A (`D - A = 0.0156` vs. `0.0347` in `v2026.10.0`).

### Documented (C1, C2, C3)
- **F4 & Committed Track A Control Battery Baseline:**
  - Documented the real held-out Track A control battery baseline (`A = 0.9607`, `B = 0.9461`, `C = 0.9829`, `D = 0.9763`, margin `D - A = 0.0156`, `E1` keyword-salad residual `0.8216`, short-query blocking `F1–F3`) and environment manifest in `README.md` (Known Limitations §1–§3) and `CONTRIBUTING.md`.

## [2026.10.0] - 2026-10-28
- Initial pre-release package (`SHA-256: 245d8327520e69a53e0dee40e17b88fcb23e84636411efb92fa43a5f8af7a53e`).
