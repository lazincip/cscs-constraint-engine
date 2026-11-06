# Benchmarks & Evaluation Protocol

## Evaluation protocol (Track A & Track B)

* **Track A (Deterministic Middleware Control):** The battery is committed to this repository (`benchmarks/track_a_battery.json`, runner `benchmarks/run_track_a.py`) and runs in two modes against the fixed 4-seed initialization corpus (`S1–S4`): **stateful** (`learn_manifold=True`) and **frozen** (`learn_manifold=False`). Since v2026.10.2 the success criteria are **friction separation** (median friction of in-domain families vs. far-OOD) and **AUROC** computed on friction, resonance deficit, and raw residual, reported together. The v2026.10.1 residual-margin criterion (raw SVD residual margin $\ge 0.10$) was **retired** in this release: the residual is geometrically floored near $\sqrt{1 - k/512} \approx 0.988$ at seed scale, so a 0.10 residual margin is unreachable by construction (see [LIMITATIONS.md](LIMITATIONS.md)); the raw residual is still reported for continuity.
* **Track B (Live SLM Statistical Track):** Reports model tag, quantization level, hardware/CPU/RAM manifest, and latency percentiles across 300 turns. Verdicts are evaluated as rates, not single runs.
* **Environment manifest (required with every submission):** OS + version · CPU model · Python version · numpy version · BLAS/LAPACK (`numpy.show_config()`) · thread env · package SHA-256 or git commit · Track A or B.

## Track A results (`v2026.10.2`)

* **Reference Environment Manifest:** `Linux VM, Python 3.12.3, numpy 1.26.4`.
* **Policy:** calibrated, $\tau_{95} = 0.6566$ (leave-one-out anchor friction, 95th percentile).
* **Allowed counts (both modes):** `A 6/6 · B 3/3 · C 4/4 · D 3/4 · E 1/3 · F 3/3 · G 2/2`. Short valid questions (`F1–F3`) pass — the v2026.10.1 brevity block is fixed. Of the `E` family, instruction-override text in fluent grammatical English (`E3`) is admitted: it is a content-policy case, out of scope for a trajectory governor (see [LIMITATIONS.md](LIMITATIONS.md)). Keyword salad (`E1`), clickbait (`E2`), and stuffed spam (`D4`) are blocked at the quality gate.
* **Median friction — stateful:** `A 0.5479 · B 0.5565 · C 0.6061 · D 0.6007 · E 0.6379 · F 0.5916 · G 0.5777` — friction margin $D - A = 0.0528$; strict ordering $A < C < D$ does **not** hold in stateful mode ($C$ exceeds $D$ by 0.0054; disclosed in [LIMITATIONS.md](LIMITATIONS.md)).
* **Median friction — frozen:** `A 0.5547 · B 0.5844 · C 0.6105 · D 0.6113 · E 0.6588 · F 0.6047 · G 0.5897` — margin $D - A = 0.0566$; ordering $A < C < D$ holds.
* **AUROC (A vs. D):** stateful — friction `0.8333`, resonance deficit `0.9167`, raw residual `0.7500`; frozen — friction `0.9167`, resonance deficit `0.9167`, raw residual `0.8333`.
* **AUROC (in-domain vs. OOD, families A+B vs. C+D+E):** friction `0.8485`, resonance deficit `0.6742`, raw residual `0.6515` (both modes).
* **Post-inference verifier pairs (`H`):** `H1 SPORADIC_RISK (1.000)`, `H2 VERIFIED (0.207)`, `H3 SPORADIC_RISK (0.947)`, `H4 SANCTIONED_DREAM (0.328)`.

### Order-dependence regression (Test K) and long-session callbacks (Test L)

* **Test K:** the demo Turn-2 prompt is **allowed at cycle 1** (friction 0.5825) **and** at cycle 4 on first presentation (friction 0.5757), resonance stable $\approx 0.456$ in both. In v2026.10.1 the same prompt was blocked at cycle 4 (friction 0.6509, resonance 0.2183) purely because of presentation order — the defect is fixed by anchor-synchronized enrollment.
* **Test L:** after 5 in-domain + 20 diverse adjacent-domain turns, exact and paraphrase callbacks to turn-1 content are allowed (friction 0.278 / 0.562) — within-domain memory survives SVD-window eviction via the mirror (32) and scout (64) stores.

### Held-out check (auditor-written prompts, never in any fixture)

12 fresh in-domain and 12 fresh far-OOD prompts (written by the independent auditor for this release, disjoint from the committed battery): friction AUROC (in vs. out) **0.8056** in both modes; median friction in-domain `0.6011`, far-OOD `0.6332` (frozen scoring). The committed battery above is a regression suite — several calibration constants in this release were validated against it — so this held-out figure is the unbiased discrimination estimate for the release.

### Session-drift isolation (300-turn probes)

Far-OOD admissions out of 30 fluent far-OOD probes, v2026.10.2: **fresh engine 29/30** (median friction 0.5835); after 300 adjacent-domain turns with learning on **29/30** (0.5936); learning off **29/30** (0.5983); mixed protocol, learning off **29/30** (0.5957). Session history moves admissions by **zero** and median friction by $\le +0.015$: there is no dynamic drift channel in this build (blocked prompts mutate nothing; boundary-band prompts commit nothing). The 29/30 figure itself is a **baseline calibration property** — the $\tau_{95}$ wall sits above most fluent far-OOD friction at 4-seed enrollment — and is documented as a limitation, not a session effect.

### Latency (300 distinct-prompt turns)

Reference VM (Linux, Python 3.12.3, numpy 1.26.4): **median 2.645 ms · mean 2.620 ms · P95 3.412 ms · max 3.574 ms**. v2026.10.2 performs additional anchor passes per turn (anchor reflect + anchor scout in the calibrated state) and is slower per turn than v2026.10.1 on the same class of hardware; cross-machine latency claims are not made.

## Committed Track A baseline (`v2026.10.1`, historical)

* **Reference Environment Manifest:** `Linux VM, Python 3.12.3, numpy 2.5.4, clean venv (pip install -e .)`. The battery prompts were not committed in that release (values only); they are committed from v2026.10.2 onward.
* **300-Turn Latency Scaling (F1 Verified):** Probe A median `1.071 ms`, P95 `1.507 ms`; Probe B median `0.887 ms`, P95 `1.369 ms`.
* **Held-Out Residual Medians:** `A 0.9607 · B 0.9461 · C 0.9829 · D 0.9763`; ordering `A < C < D` FAIL (`D < C`); residual margin $D - A = 0.0156$ vs. the (now retired) $\ge 0.10$ criterion.
* **Verifier pairs:** `H1 SPORADIC_RISK (0.865)`, `H2 VERIFIED (0.157)`, `H3 SPORADIC_RISK (0.764)`, `G1/G2 SANCTIONED_DREAM`.
* **Recorded observations:** short valid questions (`F1–F3`) and in-domain item `A4` blocked under 4-seed cold start; keyword salad (`E1`) achieved the battery's lowest residual (`0.8216`).
