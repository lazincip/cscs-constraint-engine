"""
DriftVerifier: Post-inference hallucination and semantic drift router for SLMs.
Separates explicitly invited creative brainstorming (SANCTIONED_DREAM) from
uninvited high-specificity confabulation (SPORADIC_RISK).
"""

import re
import numpy as np
from typing import Dict, Any
from .manifold import SpectralManifold


class DriftVerifier:
    """
    Classifies SLM/LLM outputs into:
    - SANCTIONED_DREAM: High novelty/exploration explicitly invited (below 0.75 risk ceiling).
    - VERIFIED: Low semantic drift, grounded or appropriately hedged claims.
    - SPORADIC_RISK: High-specificity ungrounded assertions (integers, decimals,
      years, percentages, proper names) without hedging, or extreme dream fabrication (>= 0.75).
    - AMBIGUOUS: Borderline drift requiring review.
    """

    HEDGING = {
        "might", "could", "may", "perhaps", "possibly", "probably",
        "approximately", "roughly", "suggest", "appear", "indicate",
        "hypothetically", "theoretically", "speculate", "estimate",
        "often", "although", "vary"
    }
    ASSERTIVE = {
        "always", "never", "definitely", "certainly", "proven", "proved",
        "confirmed", "established", "undeniably", "exactly", "doubt"
    }
    SPECIFICITY_RE = re.compile(r"\b(\d+(?:\.\d+)?%?|[A-Z][a-z]+ [A-Z][a-z]+)\b")
    DREAM_RISK_CEILING = 0.75

    def __init__(self, manifold: SpectralManifold):
        self.manifold = manifold

    def verify(self, prompt: str, output: str, dreaming: bool, resonance: float) -> Dict[str, Any]:
        words = set(re.findall(r"[a-z]+", output.lower()))
        n_words = max(1, len(words))
        hedge_count = len(words & self.HEDGING)
        assert_count = len(words & self.ASSERTIVE)
        hedge_ratio = hedge_count / n_words
        assert_ratio = assert_count / n_words

        specifics = len(self.SPECIFICITY_RE.findall(output))
        raw_spec_score = min(1.0, specifics / 4.0)
        hedge_discount = max(0.15, 1.0 - 0.70 * hedge_count) if assert_count == 0 else 1.0
        spec_score = raw_spec_score * hedge_discount

        vp = self.manifold.embed(prompt)
        vo = self.manifold.embed(output)
        drift = max(0.0, 1.0 - float(np.dot(vp, vo)))

        risk = (
            0.35 * spec_score +
            0.35 * drift +
            0.20 * max(0.0, assert_ratio * 10.0 - hedge_ratio * 8.0) +
            0.10 * (1.0 - resonance)
        )
        risk = max(0.0, min(1.0, risk))

        if dreaming:
            verdict = "SPORADIC_RISK" if risk >= self.DREAM_RISK_CEILING else "SANCTIONED_DREAM"
        elif risk >= 0.52:
            verdict = "SPORADIC_RISK"
        elif risk >= 0.35:
            verdict = "AMBIGUOUS"
        else:
            verdict = "VERIFIED"

        return {
            "verdict": verdict,
            "sporadic_risk": round(risk, 4),
            "semantic_drift": round(drift, 4),
            "unsolicited_specifics": specifics
        }
