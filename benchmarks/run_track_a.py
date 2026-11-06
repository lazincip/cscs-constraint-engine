"""
Standardized Track A Control Battery Runner (CONTROL_TESTS_v1.2).
Executes the committed fixture (benchmarks/track_a_battery.json) in both:
1. Stateful mode (learn_manifold=True, anchor-drift budgeted)
2. Frozen-manifold mode (learn_manifold=False)
Reports environment manifest, per-family median friction & residuals,
friction separation margin, and Mann-Whitney AUROC across transforms.
"""

import json
import os
import platform
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cscs import CSCSMiddlewareEngine, __version__


def compute_auroc(pos_scores: list, neg_scores: list) -> float:
    """Exact Mann-Whitney U statistic AUROC (probability neg_score > pos_score)."""
    if not pos_scores or not neg_scores:
        return 0.5
    wins = 0.0
    total = len(pos_scores) * len(neg_scores)
    for p in pos_scores:
        for n in neg_scores:
            if n > p:
                wins += 1.0
            elif n == p:
                wins += 0.5
    return round(wins / total, 4)


def run_battery(learn_manifold: bool = True) -> dict:
    fixture_path = os.path.join(os.path.dirname(__file__), "track_a_battery.json")
    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    engine = CSCSMiddlewareEngine()
    for _, s_text in data["seeds"].items():
        engine.manifold.learn(s_text)

    records = {}
    fam_frictions = {}
    fam_residuals = {}
    fam_inv_reson = {}

    for fam, items in data["families"].items():
        fam_frictions[fam] = []
        fam_residuals[fam] = []
        fam_inv_reson[fam] = []
        for item_id, prompt in items.items():
            drm = (fam == "G")
            pre = engine.pre_inference(prompt, dreaming=drm, learn_manifold=learn_manifold)
            records[item_id] = pre
            fam_frictions[fam].append(pre["friction"]["field_friction"])
            fam_residuals[fam].append(pre["manifold"]["reconstruction_error"])
            fam_inv_reson[fam].append(round(1.0 - pre["contraction"]["resonance"], 4))

    verifier_results = {}
    for h_id, h_spec in data["verifier_pairs"].items():
        ref_id = h_spec["prompt_ref"]
        ref_prompt = data["families"][ref_id[0]][ref_id]
        pre_rep = records[ref_id]
        if h_spec["dreaming"]:
            pre_rep = dict(pre_rep, dreaming=True)
        post = engine.post_inference(ref_prompt, h_spec["output"], pre_rep)
        verifier_results[h_id] = post

    med_fric = {k: round(float(np.median(v)), 4) for k, v in fam_frictions.items()}
    med_res = {k: round(float(np.median(v)), 4) for k, v in fam_residuals.items()}

    # In-domain positive class = A + B + F; OOD negative class = C + D + E
    in_fric = fam_frictions["A"] + fam_frictions["B"] + fam_frictions["F"]
    ood_fric = fam_frictions["C"] + fam_frictions["D"] + fam_frictions["E"]

    in_res = fam_residuals["A"] + fam_residuals["B"] + fam_residuals["F"]
    ood_res = fam_residuals["C"] + fam_residuals["D"] + fam_residuals["E"]

    in_ires = fam_inv_reson["A"] + fam_inv_reson["B"] + fam_inv_reson["F"]
    ood_ires = fam_inv_reson["C"] + fam_inv_reson["D"] + fam_inv_reson["E"]

    return {
        "mode": "stateful (learn_manifold=True)" if learn_manifold else "frozen (learn_manifold=False)",
        "policy_state": engine.policy_state,
        "wall_threshold": engine.scout.wall_threshold,
        "median_friction": med_fric,
        "median_residual": med_res,
        "friction_margin_D_minus_A": round(med_fric["D"] - med_fric["A"], 4),
        "residual_margin_D_minus_A": round(med_res["D"] - med_res["A"], 4),
        "friction_ordering_A_lt_C_lt_D": (med_fric["A"] < med_fric["C"] < med_fric["D"]),
        "auroc_A_vs_D": {
            "field_friction": compute_auroc(fam_frictions["A"], fam_frictions["D"]),
            "resonance_deficit": compute_auroc(fam_inv_reson["A"], fam_inv_reson["D"]),
            "raw_residual": compute_auroc(fam_residuals["A"], fam_residuals["D"])
        },
        "auroc_all_in_vs_ood": {
            "field_friction": compute_auroc(in_fric, ood_fric),
            "resonance_deficit": compute_auroc(in_ires, ood_ires),
            "raw_residual": compute_auroc(in_res, ood_res)
        },
        "allowed_counts": {
            fam: f"{sum(1 for i in items if records[i]['allowed'])}/{len(items)}"
            for fam, items in data["families"].items()
        },
        "verifier": verifier_results,
        "records": records
    }


if __name__ == "__main__":
    print("=" * 78)
    print(f" CSCS TRACK A CONTROL BATTERY RUNNER — v{__version__}")
    print(f" Environment: {platform.system()} {platform.release()} | Python {platform.python_version()} | NumPy {np.__version__}")
    print("=" * 78)
    for lm in (True, False):
        res = run_battery(learn_manifold=lm)
        print(f"\n--- Mode: {res['mode']} | Policy: {res['policy_state']} (tau={res['wall_threshold']}) ---")
        print("  Allowed Counts     :", res["allowed_counts"])
        print("  Median Friction    :", res["median_friction"])
        print("  Median Residual    :", res["median_residual"])
        print(f"  Friction Margin D-A: {res['friction_margin_D_minus_A']:.4f} | Ordering A<C<D: {res['friction_ordering_A_lt_C_lt_D']}")
        print("  AUROC (A vs D)     :", res["auroc_A_vs_D"])
        print("  AUROC (In vs OOD)  :", res["auroc_all_in_vs_ood"])
        print("  Verifier Verdicts  :", {k: (v["verdict"], v["sporadic_risk"]) for k, v in res["verifier"].items()})
