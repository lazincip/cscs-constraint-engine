"""
CSCSMiddlewareEngine: Unified pre-inference and post-inference governor
for hardware-constrained Small Language Models (SLMs).
"""

import time as tm
from typing import Dict, Any
from .manifold import SpectralManifold
from .contraction import BanachContractionMirror
from .scout import RBFConstraintScout
from .compression import MDLCompressionRouter
from .verifier import DriftVerifier
from .codec import T3LBStateCodec


class CSCSMiddlewareEngine:
    """
    Zero-GPU CPU Governor for Small Language Models (SLMs).
    Runs in <2.0 ms on standard CPU hardware across 300+ turn sessions to:
    1. Gate out-of-manifold / high-friction prompts before wasting SLM compute.
    2. Track Banach contraction & Lyapunov stability across multi-turn chats.
    3. Route post-inference SLM outputs into VERIFIED, SANCTIONED_DREAM, or SPORADIC_RISK.
    4. Serialize session state into a 93-byte T3LB seed for zero-token context handoffs.
    """

    def __init__(self, n_features: int = 512, max_components: int = 12):
        self.manifold = SpectralManifold(n_features=n_features, max_components=max_components)
        self.mirror = BanachContractionMirror(self.manifold)
        self.scout = RBFConstraintScout()
        self.compression = MDLCompressionRouter(self.manifold)
        self.verifier = DriftVerifier(self.manifold)
        self.codec = T3LBStateCodec()
        self.cycles = 0

    def evaluate_prompt_quality(self, prompt: str) -> float:
        words = prompt.strip().split()
        if len(words) < 3:
            return 0.20
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        len_score = min(1.0, len(words) / 12.0) if len(words) <= 150 else 0.65
        has_punct = 0.10 if any(ch in prompt for ch in ".?!,;:") else 0.0
        return round(min(1.0, 0.50 * unique_ratio + 0.40 * len_score + has_punct), 4)

    def pre_inference(self, prompt: str, dreaming: bool = False,
                      learn_manifold: bool = True) -> Dict[str, Any]:
        t0 = tm.perf_counter()
        self.cycles += 1

        quality = self.evaluate_prompt_quality(prompt)
        proj = self.manifold.project(prompt)
        ref = self.mirror.reflect(prompt)
        sc = self.scout.scout(
            proj["coords"],
            quality_score=quality,
            rec_error=proj["reconstruction_error"],
            resonance=ref["resonance"]
        )
        cmp = self.compression.compress(prompt, proj["coords"])

        blocked = (quality < 0.30) or (sc["wall_detected"] and not dreaming and self.cycles >= 3)
        if not blocked and learn_manifold:
            self.manifold.learn(prompt)

        self.codec.update_from_cycle(
            resonance=ref["resonance"],
            field_friction=sc["field_friction"],
            in_manifold=proj["in_manifold_energy"],
            quality=quality,
            blocked=blocked
        )

        latency_ms = round((tm.perf_counter() - t0) * 1000.0, 3)
        return {
            "cycle": self.cycles,
            "allowed": not blocked,
            "dreaming": dreaming,
            "quality_score": quality,
            "manifold": proj,
            "contraction": ref,
            "friction": sc,
            "compression": cmp,
            "t3lb_seed": self.codec.encode_seed(),
            "latency_ms": latency_ms
        }

    def post_inference(self, prompt: str, slm_output: str, pre_report: Dict[str, Any]) -> Dict[str, Any]:
        t0 = tm.perf_counter()
        verdict = self.verifier.verify(
            prompt=prompt,
            output=slm_output,
            dreaming=pre_report.get("dreaming", False),
            resonance=pre_report.get("contraction", {}).get("resonance", 0.5)
        )
        verdict["latency_ms"] = round((tm.perf_counter() - t0) * 1000.0, 3)
        return verdict
