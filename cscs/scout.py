"""
RBFConstraintScout: Continuous Radial Basis Function (Gaussian Kernel Density)
friction field mapper over spectral eigen-coordinates.
"""

import math as m
import numpy as np
from collections import deque
from typing import Dict, List, Any
from .manifold import EPS

PHI = (1.0 + m.sqrt(5.0)) / 2.0
INV_PHI = 1.0 / PHI


class RBFConstraintScout:
    """
    Maps constraint friction across the spectral manifold using a continuous
    Radial Basis Function (Gaussian kernel) estimator. Detects high-friction
    topological walls (field_friction > 1/Φ ≈ 0.618) prior to SLM inference.
    """

    def __init__(self, bandwidth: float = 0.45):
        self.bandwidth = bandwidth
        self.nodes: deque = deque(maxlen=64)

    def scout(self, coords: List[float], quality_score: float,
              rec_error: float, resonance: float = 0.5) -> Dict[str, Any]:
        ood_penalty = min(1.0, max(0.0, rec_error - 0.30) * 1.45)
        align_deficit = 1.0 - max(0.0, min(1.0, resonance))
        local_friction = min(
            1.0,
            0.30 * (1.0 - quality_score) + 0.45 * ood_penalty + 0.25 * align_deficit
        )

        if not coords:
            return {
                "field_friction": round(local_friction, 4),
                "local_friction": round(local_friction, 4),
                "wall_detected": local_friction > INV_PHI,
                "kernel_density": 0.0
            }

        c_vec = np.array(coords, dtype=np.float64)
        if not self.nodes:
            self.nodes.append((c_vec, local_friction))
            return {
                "field_friction": round(local_friction, 4),
                "local_friction": round(local_friction, 4),
                "wall_detected": local_friction > INV_PHI,
                "kernel_density": 1.0
            }

        dim = len(c_vec)
        inv_2bw2 = 1.0 / (2.0 * (self.bandwidth ** 2))
        w_sum = 0.0
        wf_sum = 0.0
        for past_vec, past_f in self.nodes:
            d = min(dim, len(past_vec))
            diff = c_vec[:d] - past_vec[:d]
            dist_sq = float(np.dot(diff, diff))
            w = m.exp(-dist_sq * inv_2bw2)
            w_sum += w
            wf_sum += w * past_f

        avg_density = w_sum / len(self.nodes)

        alpha = 0.0 if rec_error >= 0.95 else min(0.35, avg_density * 0.5)
        if w_sum > EPS:
            kde_friction = wf_sum / w_sum
            field_friction = (1.0 - alpha) * local_friction + alpha * kde_friction
        else:
            field_friction = local_friction

        self.nodes.append((c_vec, local_friction))
        return {
            "field_friction": round(field_friction, 4),
            "local_friction": round(local_friction, 4),
            "wall_detected": field_friction > INV_PHI,
            "kernel_density": round(avg_density, 4)
        }
