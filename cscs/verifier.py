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
    - SANCTIONED_DREAM: High novelty/exploration explicitly invited.
    - VERIFIED: Low semantic drift, grounded claims.
    - SPORADIC_RISK: High-specificity ungrounded assertions (dates, percentages,
      proper names) without hedging when dream mode is OFF.
    - AMBIGUOUS: Borderline drift requiring review.
    """

    HEDGING = {
        "might", "could", "may", "perhaps", "possibly", "probably",
        "approximately", "roughly", "suggest", "appear", "indicate",
        "hypothetically", "theoretically", "speculate", "estimate"
    }
    ASSERTIVE = {
        "always", "never", "definitely", "certainly", "proven",
        "confirmed", "established", "undeniably", "exactly"
    }
    SPECIFICITY_RE = re.compile(r"\b(\d{4}|\d+\.\d+%?|[A-Z][a-z]+ [A-Z][a-z]+)\b")

    def __init__(self, manifold: SpectralManifold):
        self.manifold = manifold

    def verify(self, prompt: str, output: str, dreaming: bool, resonance: float) -> Dict[str, Any]:
        if dreaming:
            return {
                "verdict": "SANCTIONED_DREAM",
                "sporadic_risk": 0.0,
                "semantic_drift": round(
                    1.0 - float(np.dot(self.manifold.embed(prompt), self.manifold.embed(output))), 4
                ),
                "unsolicited_specifics": 0
            }

        words = set(re.findall(r"[a-z]+", output.lower()))
        n_words = max(1, len(words))
        hedge_ratio = len(words & self.HEDGING) / n_words
        assert_ratio = len(words & self.ASSERTIVE) / n_words

        specifics = len(self.SPECIFICITY_RE.findall(output))
        spec_score = min(1.0, specifics / 4.0)

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

        if risk >= 0.52:
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
