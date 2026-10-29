"""
BanachContractionMirror: Models context alignment as an iterated contraction
mapping on the unit hypersphere S^{F-1} to stabilize Small Language Model (SLM)
multi-turn attention drift without consuming context window tokens.
"""

import math as m
import numpy as np
from collections import deque
from typing import Dict, Any
from .manifold import SpectralManifold, EPS

PHI = (1.0 + m.sqrt(5.0)) / 2.0


class BanachContractionMirror:
    """
    Given prompt embedding x_0 and context attractors C, applies the operator:
        x_{t+1} = normalize((1 - L_t) * x_t + L_t * A(x_t, C))
    where A(x_t, C) is the Boltzmann barycenter of memory vectors and
    L_t = Φ^{-t} guarantees contraction (Lipschitz < 1 for t >= 1).
    """

    def __init__(self, manifold: SpectralManifold, temp: float = 0.15, max_memory: int = 32):
        self.manifold = manifold
        self.temp = temp
        self.memory_vectors: deque = deque(maxlen=max_memory)

    def reflect(self, prompt: str, max_iter: int = 8, tol: float = 1e-4) -> Dict[str, Any]:
        x0 = self.manifold.embed(prompt)
        if not self.memory_vectors:
            self.memory_vectors.append(x0)
            return {
                "iterations": 0,
                "resonance": 1.0,
                "lyapunov_exponent": 0.0,
                "contraction_displacement": 0.0,
                "converged": True
            }

        C = np.array(self.memory_vectors, dtype=np.float64)
        xt = x0.copy()
        displacements = []

        for t in range(1, max_iter + 1):
            sims = C @ xt
            weights = np.exp((sims - float(np.max(sims))) / self.temp)
            weights /= (float(np.sum(weights)) + EPS)
            attractor = weights @ C
            attr_norm = float(np.sqrt(np.dot(attractor, attractor)))
            if attr_norm > EPS:
                attractor /= attr_norm

            lip = PHI ** (-t)
            x_next = (1.0 - lip) * xt + lip * attractor
            x_next_norm = float(np.sqrt(np.dot(x_next, x_next)))
            if x_next_norm > EPS:
                x_next /= x_next_norm

            diff = x_next - xt
            step_dist = float(np.sqrt(np.dot(diff, diff)))
            displacements.append(step_dist)
            xt = x_next
            if step_dist < tol:
                break

        max_direct_sim = float(np.max(C @ x0))
        fp_sim = float(np.dot(x0, xt))
        resonance = max(0.0, min(1.0, 0.5 * max_direct_sim + 0.5 * fp_sim))
        tot_diff = xt - x0
        total_disp = float(np.sqrt(np.dot(tot_diff, tot_diff)))
        self.memory_vectors.append(x0)

        if len(displacements) >= 2 and displacements[0] > EPS:
            ratios = [
                m.log((displacements[i] + EPS) / (displacements[i - 1] + EPS))
                for i in range(1, len(displacements))
            ]
            lyapunov = float(np.mean(ratios))
        else:
            lyapunov = -m.log(PHI)

        return {
            "iterations": len(displacements),
            "resonance": round(resonance, 4),
            "lyapunov_exponent": round(lyapunov, 4),
            "contraction_displacement": round(total_disp, 4),
            "converged": displacements[-1] < tol if displacements else True
        }
