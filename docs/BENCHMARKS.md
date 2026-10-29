# Benchmarks & Evaluation Protocol

## Evaluation protocol (Track A & Track B)

All benchmark submissions must adhere to the standardized control protocol (CONTROL_TESTS_v1), established by independent verification (Agent CPh, 2026-10-28):

* **Track A (Deterministic Middleware Control):** Uses the fixed 4-seed domain initialization corpus (`S1–S4`) and a fixed held-out battery (`A–H`) run in order; evaluates relative within-run ordering (`median_residual(in-domain) < median_residual(near-neighbor) < median_residual(far-OOD)` with separation margin $\ge 0.10$) and exact canned-output verdict agreement (`VERIFIED`, `SPORADIC_RISK`, `SANCTIONED_DREAM`). No live SLM — the model is removed from the measurement.
* **Track B (Live SLM Statistical Track):** Reports model tag, quantization level, hardware/CPU/RAM manifest, and latency percentiles across 300 turns. Verdicts are evaluated as rates, not single runs.
* **Environment manifest (required with every submission):** OS + version · CPU model · Python version · numpy version · BLAS/LAPACK (`numpy.show_config()`) · thread env · package SHA-256 or git commit · Track A or B.

## Committed Track A baseline (`v2026.10.1`)

* **Reference Environment Manifest:** `Linux VM, Python 3.12.3, numpy 2.5.4, clean venv (pip install -e .)`
* **300-Turn Latency Scaling Baseline (F1 Verified):**
  * **Probe A (CPh 300-turn probe):** Turn 5 / 50 / 100 / 200 / 300 = `1.288 / 1.041 / 0.851 / 1.102 / 1.074 ms` | **Median:** `1.071 ms` | **Mean:** `1.148 ms` | **P95:** `1.507 ms` | **Max:** `24.8 ms` (single outlier spike; $\sim 130\times$ improvement over `v2026.10.0` at turn 300).
  * **Probe B (Argon prompt style):** **Median:** `0.887 ms` | **P95:** `1.369 ms` | **Max:** `1.768 ms`.
  * **Component Profile at Turn 200:** `project: 0.03 ms` | `reflect: 0.32 ms` | `scout: 0.18 ms` | `compress: 0.11 ms` | `learn: 0.00 ms`.
* **Track A Held-Out Control Battery (`S1–S4` seeds, then battery `A–H` in order via `pre_inference`, `H` via `post_inference`):**
  * **Held-Out Residual Medians (`v2026.10.1` vs. `v2026.10.0`):**
    * `A` (Held-out in-domain): **`0.9607`** (`v2026.10.0`: `0.9267`)
    * `B` (Paraphrase): **`0.9461`**
    * `C` (Near-neighbor technical): **`0.9829`** (`v2026.10.0`: `0.9629`)
    * `D` (Far-OOD): **`0.9763`** (`v2026.10.0`: `0.9614`)
  * **Ordering & Margin Status:** Ordering `A < C < D` = **FAIL** (`D < C`); separation margin $D - A = 0.0156$ (`v2026.10.0`: $0.0347$), below target criterion $\ge 0.10$ (documented cold-start calibration target F4 — see [LIMITATIONS.md](LIMITATIONS.md)).
  * **Residual Normalization Bounds (F3):** Battery max residual = **`0.9949 <= 1.0`** (**PASS**).
  * **Post-Inference Verifier Pairs (`H` & `G`):**
    * `H1`: `SPORADIC_RISK` (`risk = 0.865`), `H2`: `VERIFIED` (`risk = 0.157`), `H3`: `SPORADIC_RISK` (`risk = 0.764`), `G1/G2`: `SANCTIONED_DREAM` (**PASS** — 4-decimal parity with `v2026.10.0`).
  * **Recorded Protocol Observations:** Short valid questions (`F1–F3`, $< 3$ words) and held-out in-domain item `A4` are blocked under 4-seed cold start; domain keyword salad (`E1`) achieves the lowest residual of the battery (`0.8216` in `v2026.10.1`; `0.7623` in `v2026.10.0`).
