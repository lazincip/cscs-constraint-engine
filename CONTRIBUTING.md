# Contributing & Open Call for SLM Benchmarking Collaborators

We actively welcome systems engineers, local AI builders, and researchers working on **Small Language Models (SLMs)**, **Edge AI**, and **Deterministic AI Telemetry** to test, benchmark, and expand the **CSCS Constraint Physics Engine**.

---

## High-Priority Collaboration Tracks

1. **Cold-Start Subspace Calibration & Keyword-Salad Defense (F4 / Track A Challenge):**
   * Improving held-out in-domain vs. far-OOD separation margin ($D - A \ge 0.10$ and $A < C < D$ ordering) under small-seed initialization ($N = 4$) while resisting n-gram keyword-salad gaming (`E1`) and preserving $< 2.0\text{ ms}$ P95 CPU latency.
2. **SLM Runtime Integrations (`llama.cpp`, `Ollama`, `vLLM`, `MLX`):**
   * Building native pre/post-inference hooks for Apple Silicon (`MLX` / unified memory), Raspberry Pi / Jetson edge nodes, and quantized 1B–8B parameter models (`Llama-3.2`, `Qwen-2.5`, `Phi-4`, `Gemma-2`).
3. **Empirical Benchmark Suites (`HalluEval`, `TruthfulQA`, `LongBench`):**
   * Measuring compute saved (blocked OOD/low-quality turns) and precision/recall of `DriftVerifier` (`SPORADIC_RISK` vs. `VERIFIED`) across quantized 4-bit and 8-bit SLMs compared to heavy neural guardrails (e.g., LlamaGuard).
4. **Hardware Telemetry & Energy Auditing:**
   * Wiring real-time hardware power counters (`powermetrics` on macOS, `NVML` on Linux) into the state governor to measure joules-per-token saved by pre-inference manifold gating.

---

## Evaluation Protocol & Committed Baseline

All benchmark submissions must adhere to the standardized control protocol (Track A / Track B) and are compared against the **committed `v2026.10.1` baseline** — full protocol, environment manifest, latency probes, and held-out battery results live in [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Known limitations the baseline documents are in [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

---

## How to Submit Benchmarks or Pull Requests

1. Fork the repository and create a feature or benchmark branch (`git checkout -b bench/qwen-3b-truthfulqa`).
2. Run the mathematical invariant test suite to ensure zero regressions:
   ```bash
   python -m unittest discover -s tests -v
   ```
3. Open a Pull Request or start a thread in **GitHub Discussions** with your environment manifest, Track A / Track B logs, and latency percentiles.

All contributors whose benchmarks or modules are merged will be credited in the release notes and project documentation, while core architectural priority remains protected under **Apache License 2.0** (`LICENSE` + `NOTICE`) and `CITATION.cff`.
