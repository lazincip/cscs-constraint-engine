# Changelog

All notable changes to `cscs-constraint-engine` are documented in this file.

## [2026.10.2] - 2026-11-06 (Addendum Round: External-Audit Fixes)

This release implements the consolidated fix queue from five independent reviews of v2026.10.1 (a preregistered self-benchmark study; code-level critiques by Qwen and Copilot; from-scratch reproduction by Mistral; execution audits by Maya/Sesame AI and Z.ai/GLM; synthesis by DeepSeek), adjudicated by the independent auditor (Agent CPh) and decided by the PI. Per the PI's record requirement, **everything dropped or modified in this round is recorded here**, including documentation-only decisions.

### Fixed
- **Presentation-order block (Test K / DeepSeek H1).** A prompt's verdict no longer depends on when it is first presented. Root cause: enrollment left the contraction mirror's memory empty, so early resonance was synthetic (identically 1.0) under a cycles<3 grace period. Enrollment now anchors the mirror, the grace period is removed, and resonance is anchor-weighted (0.70 anchor barycenter / 0.30 session trajectory). The demo Turn-2 prompt is allowed at cycle 1 (friction 0.5825) and at cycle 4 on first presentation (0.5757); in v2026.10.1 it blocked at cycle 4 (0.6509).
- **State poisoning.** Blocked prompts previously wrote mirror memory and scout nodes before the gate decided. Blocked prompts now mutate no module state; additionally, in the calibrated state, allowed prompts commit state (mirror, scout, compression, manifold) only from the core attractor zone (anchor-only friction ≤ 1/Φ) — the two-zone commit budget. Covered by a new regression test (`test_state_poisoning_and_terse_salad_gates`).
- **Numerical-noise principal axis.** `SpectralManifold._fit_subspace` used an absolute singular-value threshold (S > 1e-9) that admitted a noise axis at the centered-rank limit (4th SV ≈ 1.6e-8 at n = 4). Replaced with a relative eigenvalue cutoff (λ ≥ 1e-12 × trace(G)); verified k = 3 / 4 / 9 at n = 4 / 5 / 10, covered by a new invariant test.
- **Brevity penalty.** Terse valid prompts (e.g. "Explain UCEs.") were blocked 93% of the time in the benchmark study. The quality gate was rebuilt (content-word uniqueness, reduced length saturation, stopword-capped stuffing penalty): battery items F1–F3 pass, while keyword salad (E1), clickbait (E2), and stuffed spam (D4) are blocked at the quality gate.
- **Verifier integer blindness.** `SPECIFICITY_RE` matched almost no plain integers; the demo's correct reply ("58 to 63 Ma") counted zero specifics. The pattern set now covers plain integers, integer percentages, sentence-initial specifics, and unit quantities, and hedged citations are discounted (hedge-discount slope 0.70, floor 0.15, calibrated so hedged integer ranges remain VERIFIED).
- **Dream-mode scoring.** `DriftVerifier.verify` in dream mode previously returned a constant risk (0.05), exempting dream outputs from audit. Dream replies are now scored with the standard risk and hard-capped at 0.75. Wall bypass under `dreaming=True` is unchanged (documented policy; dreaming never alters any score, only the consequence at the wall).
- **T3LB codec.** All 13 state dimensions now update from live telemetry every cycle (7 of 13 were constants). `decode()` round-trips at 2-decimal precision (new invariant test, including the BLAKE2b empty-string commitment vector).
- **Penalty-transform saturation.** The OOD penalty `(r − 0.30) × 1.45` saturated for r ≥ ~0.99; replaced with `(r − 0.30) / 0.70` in the scout, and the same transform harmonized into the graduation (τ95) path so calibration and live scoring use one friction definition.

### Changed
- **Policy: hybrid graduation (PI decision Q2 = D2 + automatic shift).** The engine starts adaptive (fixed 1/Φ wall, cold-start anchor accumulation) and graduates automatically to a calibrated wall τ95 (leave-one-out 95th-percentile anchor friction) once τ95 is estimable from ≥ 4 anchored mirror vectors. Reference battery: **τ95 = 0.6566**. `pre_inference` results now report `policy_state` and the active `wall_threshold`.
- **Evaluation criteria (PI decision Q1).** Track A success is now judged by **friction separation + AUROC** (friction, resonance deficit, and raw residual reported together). The v2026.10.1 criterion — raw-residual margin ≥ 0.10 with strict A < C < D residual ordering — is **retired**: the residual floor (≈ √(1 − k/512) ≈ 0.988) makes a 0.10 residual margin geometrically unreachable at seed scale. Project description reframed drift-first (trajectory discipline); the project title is unchanged.
- **Demo fingerprint.** Demo Turn-1 post-inference risk: **0.195 (v2026.10.1) → 0.295 (v2026.10.2)**, a consequence of the verifier integer fix above. All other demo verdicts are unchanged (VERIFIED / SPORADIC_RISK / SANCTIONED_DREAM / BLOCKED).
- **Benchmark fixture committed.** The Track A battery (25 prompts + 4 verifier pairs) and its runner are now in the repository (`benchmarks/`) — in v2026.10.1 only the resulting values were published, which external audit correctly flagged as not reproducible. Note: several calibration constants in this release were validated against this battery; it functions as a regression suite, and the release's unbiased discrimination estimate is the auditor's held-out check (AUROC 0.8056) in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
- **Latency.** The calibrated path performs additional anchor passes per turn; measured on the reference VM (Linux, Python 3.12.3, numpy 1.26.4): 300 distinct-prompt turns, median 2.645 ms / P95 3.412 ms. v2026.10.1 figures (0.45–1.07 ms medians, other hardware) do not carry over; the README states the operating envelope with its environment instead of a cross-machine figure.
- **Metadata.** `CITATION.cff` now carries the **concept DOI** (10.5281/zenodo.23750262), chosen deliberately for release-over-release stability (v2026.10.1's file carried no DOI line; its README badge referenced the version DOI). Version bumped to 2026.10.2 in `pyproject.toml`, `CITATION.cff`, and `cscs/__init__.py`.

### Documented (no code change)
- **Intended use & non-goals** statement added to README (PI's audience ruling): the engine serves known, deliberately seeded domains on operator hardware; it is not an anything-goes chat guardrail.
- **Nomenclature note** added to README: CSCS = Constraint Space Coordinate System (Yamagata University), unrelated to the Swiss National Supercomputing Centre (CSCS, Lugano).
- **CJK limitation** documented: unspaced scripts score as single-token prompts at the quality gate (0.20) and are blocked; a tokenizer fix is scoped separately.
- **Verifier and Lyapunov scope notes** added to [docs/LIMITATIONS.md](docs/LIMITATIONS.md): the verifier scores specificity density, not truth; the Lyapunov exponent is negative by construction.
- **Session-drift isolation results** published in [docs/BENCHMARKS.md](docs/BENCHMARKS.md): no dynamic drift channel (fresh vs. 300-turn admissions identical); the calibrated wall's baseline permissiveness at 4 seeds (29/30 fluent far-OOD admitted) is documented as Limitation 1 rather than attributed to drift.
- **State restore** (`load_seed`) remains unimplemented and is now documented as such.

### Verification (independent auditor, this release)
Unit tests 6/6 · demo verdicts and fingerprints reproduced · Track A both modes (stateful margin 0.0528, AUROC A-vs-D friction 0.8333 / resonance 0.9167; frozen margin 0.0566, AUROC 0.9167 / 0.9167) · Test K and Test L pass · SV spectrum exact at n = 4/5/10 · held-out AUROC 0.8056 · drift isolation arms equal · latency as above.

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
