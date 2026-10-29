"""
Unit tests verifying mathematical invariants of the CSCS Constraint Physics Engine:
1. Orthonormality & deterministic sign-canonicalization of SVD basis vectors (F2).
2. Centered reconstruction residual strictly bounded in [0.0, 1.0] (F3).
3. Banach contraction convergence and negative Lyapunov exponent.
4. Exact regex boundary detection in DriftVerifier.
5. Lossless bi-directional serialization of T3LB state seeds at 2-decimal precision (F6).
"""

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

    def test_svd_orthonormality(self):
        """Verify that learned SVD components V_k form a sign-canonicalized orthonormal basis: V_k @ V_k^T = I."""
        V = self.manifold.components
        self.assertIsNotNone(V)
        k = V.shape[0]
        gram = V @ V.T
        np.testing.assert_allclose(gram, np.eye(k), atol=1e-6)
        for i in range(k):
            max_idx = int(np.argmax(np.abs(V[i])))
            self.assertGreaterEqual(V[i, max_idx], 0.0)

    def test_in_manifold_vs_ood_residual(self):
        """Verify in-manifold prompts have near-zero residual while OOD prompts spike within [0.0, 1.0]."""
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

    def test_verifier_regex_word_boundaries(self):
        """Verify DriftVerifier catches sentence-initial and punctuation-adjacent specifics."""
        verifier = DriftVerifier(self.manifold)
        hallucinated = "2019 was the year Dr. James Waverly proved 94.7% collapse at Stanford University."
        res = verifier.verify(self.corpus[0], hallucinated, dreaming=False, resonance=0.35)
        self.assertGreaterEqual(res["unsolicited_specifics"], 3)
        self.assertEqual(res["verdict"], "SPORADIC_RISK")

        dream_res = verifier.verify(self.corpus[0], hallucinated, dreaming=True, resonance=0.35)
        self.assertEqual(dream_res["verdict"], "SANCTIONED_DREAM")

    def test_t3lb_seed_roundtrip(self):
        """Verify T3LBStateCodec encodes and decodes 13 dimensions losslessly at 2-decimal precision."""
        codec = T3LBStateCodec()
        seed = codec.encode_seed("v2026.10")
        decoded = T3LBStateCodec.decode_seed(seed)
        self.assertEqual(decoded["version"], "v2026.10")
        self.assertEqual(len(decoded["dimensions"]), 13)
        for k, v in codec.dims.items():
            self.assertAlmostEqual(decoded["dimensions"][k], round(v, 2), places=2)


if __name__ == "__main__":
    unittest.main()
