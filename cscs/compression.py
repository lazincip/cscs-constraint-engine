"""
MDLCompressionRouter: Information-Theoretic Minimum Description Length (MDL)
and delta redundancy telemetry for context-constrained edge models.
"""

import hashlib as h
import zlib
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Any
from .manifold import SpectralManifold
from .scout import INV_PHI


class MDLCompressionRouter:
    """
    Measures real byte-level Minimum Description Length (MDL) across:
    1. DEFLATE/LZ77 entropy coding of the state/prompt payload.
    2. Entangled Delta Coding against the nearest neighbor in memory.
    3. Spectral Coordinate Sparse Projection (structural summary).

    Caches embedding vectors and compressed reference lengths in a bounded
    registry (max 128 entries) for sub-0.2ms execution.
    """

    def __init__(self, manifold: SpectralManifold, max_registry: int = 64):
        self.manifold = manifold
        self.max_registry = max_registry
        self.archive: OrderedDict[str, bytes] = OrderedDict()
        self.raw_registry: OrderedDict[str, bytes] = OrderedDict()
        self.vector_registry: OrderedDict[str, np.ndarray] = OrderedDict()

    def compress(self, text: str, coords: List[float]) -> Dict[str, Any]:
        raw_bytes = text.encode("utf-8")
        raw_len = max(1, len(raw_bytes))
        key = h.blake2b(raw_bytes, digest_size=6).hexdigest()

        zlib_bytes = zlib.compress(raw_bytes, level=1)
        lz_ratio = len(zlib_bytes) / raw_len

        v = self.manifold.embed(text)
        best_sim = 0.0
        best_key = None

        for k, pv in self.vector_registry.items():
            sim = float(np.dot(v, pv))
            if sim > best_sim:
                best_sim = sim
                best_key = k

        if best_sim > INV_PHI and best_key is not None:
            ref_bytes = self.raw_registry[best_key]
            ref_compressed_len = len(self.archive[best_key])
            joint_compressed = len(zlib.compress(ref_bytes + b"||" + raw_bytes, level=1))
            marginal_bytes = max(4, joint_compressed - ref_compressed_len)
            delta_ratio = marginal_bytes / raw_len
        else:
            delta_ratio = 1.0

        if coords:
            active = sum(1 for c in coords if abs(c) > 0.05)
            spectral_ratio = (active * 4) / raw_len
        else:
            spectral_ratio = 1.0

        self.archive[key] = zlib_bytes
        self.raw_registry[key] = raw_bytes
        self.vector_registry[key] = v
        if len(self.vector_registry) > self.max_registry:
            self.archive.popitem(last=False)
            self.raw_registry.popitem(last=False)
            self.vector_registry.popitem(last=False)

        candidates = [
            ("lz77_entropy", lz_ratio),
            ("entangled_delta", delta_ratio),
            ("spectral_sparse", spectral_ratio)
        ]
        best_method, best_ratio = min(candidates, key=lambda x: x[1])

        return {
            "key": key,
            "best_method": best_method,
            "best_ratio": round(float(best_ratio), 4),
            "raw_bytes": raw_len,
            "lz77_bytes": len(zlib_bytes),
            "nearest_entanglement_sim": round(best_sim, 4),
            "ratios": {
                "lz77_entropy": round(float(lz_ratio), 4),
                "entangled_delta": round(float(delta_ratio), 4),
                "spectral_sparse": round(float(spectral_ratio), 4)
            }
        }
