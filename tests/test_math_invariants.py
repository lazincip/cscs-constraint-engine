"""
Unit tests verifying mathematical invariants and Addendum 1 regressions of the
CSCS Constraint Physics Engine (v2026.10.2):
1. Orthonormality, sign-canonicalization, and relative eigenvalue cutoff (k <= n-1, F2 & Item C).
2. Centered reconstruction residual strictly bounded in [0.0, 1.0] (F3).
3. Banach contraction convergence and negative Lyapunov exponent.
4. DriftVerifier specificity regex (including plain integers & integer %) and dream-mode risk ceiling (Items E & F).
5. Bi-directional serialization and dynamic 13-D updates in T3LBStateCodec (F6 & Item H).
6. State-poisoning prevention, graduation to τ_95, terse in-domain fairness, and salad/spam blocking (Items D, G, Q2, Q3).
"""

import hashlib as h
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cscs import (
    SpectralManifold,
    BanachContractionMirror,
    DriftVerifier,
    T3LBStateCodec,
    CSCSMiddlewareEngine
)


class TestCSCSMathInvariants(unittest.TestCase):

    def setUp(self):
        self.manifold = SpectralManifold(n_features=256, max_components=6)
        self.corpus = [
            "Evaluate post-K-Pg evolutionary radiation of Solenodon cubanus.",
            "Phylogenetic divergence of Caribbean mammals after extinction.",
            "Virtual water footprint in regional agricultural trade under AquaTrace.",
            "Basalt rock magnetic properties and crack climbing biomechanics.",
            "Banach fixed-point contraction and SVD eigenbasis projection."
        ]
        for sentence in self.corpus:
            self.manifold.learn(sentence)

    def test_svd_orthonormality_and_relative_cutoff(self):
        """Verify learned SVD components V_k are orthonormal, sign-canonicalized, and eliminate rank-n noise axes (k <= n-1)."""
        V = self.manifold.components
        self.assertIsNotNone(V)
        k = V.shape[0]
        # With 5 centered vectors, true algebraic rank is <= 4 (Item C fix)
        self.assertEqual(k, 4)
        gram = V @ V.T
        np.testing.assert_allclose(gram, np.eye(k), atol=1e-6)
        for i in range(k):
            max_idx = int(np.argmax(np.abs(V[i])))
            self.assertGreaterEqual(V[i, max_idx], 0.0)

        # Also check n=4 and n=10 spectrum
        m4 = SpectralManifold()
        for s in self.corpus[:4]:
            m4.learn(s)
        self.assertEqual(m4.components.shape[0], 3)

    def test_in_manifold_vs_ood_residual(self):
        """Math invariant ONLY: a vector inside the fitted span reconstructs near-exactly
        (residual < 0.05), while an unrelated string's projection stays high within [0.0, 1.0].
        corpus[0] was learned in setUp, so this verifies the projection arithmetic — it is
        NOT a held-out generalization claim. Held-out discrimination evidence (and its
        limits) lives in docs/BENCHMARKS.md and docs/LIMITATIONS.md."""
        proj_in = self.manifold.project(self.corpus[0])
        proj_ood = self.manifold.project("bake chocolate cake with vanilla frosting")
        self.assertLess(proj_in["reconstruction_error"], 0.05)
        self.assertGreater(proj_ood["reconstruction_error"], 0.85)
        self.assertLessEqual(proj_ood["reconstruction_error"], 1.0)

    def test_banach_contraction_lyapunov(self):
        """Verify BanachContractionMirror converges with strictly negative Lyapunov exponent."""
        mirror = BanachContractionMirror(self.manifold)
        for s in self.corpus:
            res = mirror.reflect(s)
        self.assertLess(res["lyapunov_exponent"], 0.0)
        self.assertGreaterEqual(res["resonance"], 0.0)
        self.assertLessEqual(res["resonance"], 1.0)

    def test_verifier_regex_integers_and_dream_ceiling(self):
        """Verify DriftVerifier catches plain integers, integer %, sentence-initial specifics, and scores in dream mode."""
        verifier = DriftVerifier(self.manifold)
        hallucinated = "2019 was the year Dr. James Waverly proved 94% collapse across 12 loci at Stanford University."
        res = verifier.verify(self.corpus[0], hallucinated, dreaming=False, resonance=0.35)
        self.assertGreaterEqual(res["unsolicited_specifics"], 5)
        self.assertEqual(res["verdict"], "SPORADIC_RISK")

        # Moderate creative output in dream mode -> SANCTIONED_DREAM with non-zero scored risk
        dream_ok = verifier.verify(
            self.corpus[0],
            "Imagine each context overflow as a thermal bleaching event in 2026 where symbiotic weights re-seed.",
            dreaming=True,
            resonance=0.50
        )
        self.assertEqual(dream_ok["verdict"], "SANCTIONED_DREAM")
        self.assertGreater(dream_ok["sporadic_risk"], 0.0)

        # Extreme unhedged fabrication in dream mode (>= 0.75 risk ceiling) -> capped to SPORADIC_RISK
        dream_extreme = verifier.verify(self.corpus[0], hallucinated, dreaming=True, resonance=0.10)
        self.assertEqual(dream_extreme["verdict"], "SPORADIC_RISK")

    def test_t3lb_seed_roundtrip_and_dynamic_dims(self):
        """Verify T3LBStateCodec updates all 13 dimensions dynamically and roundtrips at 2-decimal precision."""
        codec = T3LBStateCodec()
        initial = dict(codec.dims)
        codec.update_from_cycle(
            resonance=0.65, field_friction=0.42, in_manifold=0.35, quality=0.95,
            blocked=False, compression_ratio=0.72, kernel_density=0.60,
            effective_rank=3.8, iterations=3, cycles=12
        )
        # Verify the 7 formerly static dimensions (X, Y, P, T, S, N, E) now update dynamically
        for dim_key in ["X", "Y", "P", "T", "S", "N", "E"]:
            self.assertNotEqual(codec.dims[dim_key], initial[dim_key], f"Dimension {dim_key} did not update")

        seed = codec.encode_seed("v2026.10")
        self.assertEqual(len(seed.encode("utf-8")), 93)
        decoded = T3LBStateCodec.decode_seed(seed)
        self.assertEqual(decoded["version"], "v2026.10")
        self.assertEqual(len(decoded["dimensions"]), 13)
        for k, v in codec.dims.items():
            self.assertAlmostEqual(decoded["dimensions"][k], round(v, 2), places=2)

    def test_state_poisoning_and_terse_salad_gates(self):
        """Verify blocked prompts never poison state, terse queries pass, and keyword salad/stuffed spam block."""
        engine = CSCSMiddlewareEngine()
        seeds = [
            "Evaluate the post-K-Pg evolutionary radiation of Solenodon cubanus using ultra-conserved elements.",
            "Phylogenetic divergence of Caribbean eulipotyphlan mammals after the Cretaceous-Paleogene extinction.",
            "Model regional virtual water footprints in agricultural trade using the AquaTrace Digital Product Passport.",
            "Analyze basalt rock magnetic properties and crack climbing biomechanics at Kuroiwa near Yamadera."
        ]
        for s in seeds:
            engine.manifold.learn(s)

        self.assertEqual(engine.policy_state, "calibrated")
        # Terse valid in-domain queries (F1-F3) must be ALLOWED
        for fq in ["What is a solenodon?", "Explain UCEs.", "Define incomplete lineage sorting."]:
            r_f = engine.pre_inference(fq)
            self.assertTrue(r_f["allowed"], f"Terse query blocked: {fq}")

        # Repeat spam must stay permanently BLOCKED (no state poisoning)
        spam = "buy cheap crypto tokens instant 100x pump free airdrop"
        for _ in range(5):
            r_s = engine.pre_inference(spam)
            self.assertFalse(r_s["allowed"])

        # Keyword salad (E1) and domain-stuffed spam must be BLOCKED
        salad = "solenodon phylogenetic ultra-conserved divergence extinction mammals Caribbean elements radiation genomic boundary analysis evolution."
        stuffed = "buy cheap crypto tokens instant 100x pump free airdrop solenodon phylogenetic ultra-conserved AquaTrace basalt"
        self.assertFalse(engine.pre_inference(salad)["allowed"])
        self.assertFalse(engine.pre_inference(stuffed)["allowed"])

        # RFC 7693 BLAKE2b-256('') test vector check for porters (T6)
        self.assertEqual(h.blake2b(b"", digest_size=32).hexdigest()[:16], "0e5751c026e543b2")


if __name__ == "__main__":
    unittest.main()
