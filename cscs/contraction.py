"""
BanachContractionMirror: Models context alignment as an iterated contraction
mapping on the unit hypersphere S^{F-1} to stabilize Small Language Model (SLM)
multi-turn attention drift without consuming context window tokens.
"""

import math as m
import re
import numpy as np
from collections import deque
from typing import Dict, List, Any
from .manifold import SpectralManifold, EPS

PHI = (1.0 + m.sqrt(5.0)) / 2.0
_WORD_RE = re.compile(r"[a-z0-9_]+")


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
        self.anchors: List[np.ndarray] = []
        self.max_anchors: int = 12
        self.memory_vectors: deque = deque(maxlen=max_memory)

    def add_anchor(self, vec: np.ndarray):
        if len(self.anchors) < self.max_anchors:
            if not any(float(np.dot(vec, a)) > 1.0 - 1e-9 for a in self.anchors):
                self.anchors.append(vec)
        if not any(float(np.dot(vec, m_v)) > 1.0 - 1e-9 for m_v in self.memory_vectors):
            self.memory_vectors.append(vec)

    def _active_attractors(self, anchors_only: bool = False) -> np.ndarray:
        if anchors_only and self.anchors:
            return np.array(self.anchors, dtype=np.float64)
        pool = list(self.anchors)
        for m_v in self.memory_vectors:
            if not any(float(np.dot(m_v, a)) > 1.0 - 1e-9 for a in pool):
                pool.append(m_v)
        return np.array(pool, dtype=np.float64) if pool else np.empty((0, self.manifold.n_features))

    def reflect(self, prompt: str, max_iter: int = 8, tol: float = 1e-4,
                mutate: bool = True, anchors_only: bool = False) -> Dict[str, Any]:
        x0 = self.manifold.embed(prompt)
        C = self._active_attractors(anchors_only=anchors_only)
        if C.shape[0] == 0:
            if mutate:
                self.memory_vectors.append(x0)
            return {
                "iterations": 0,
                "resonance": 1.0,
                "lyapunov_exponent": 0.0,
                "contraction_displacement": 0.0,
                "converged": True
            }

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

        if self.anchors:
            A_mat = np.array(self.anchors, dtype=np.float64)
            a_sims = A_mat @ x0
            max_direct_sim = float(np.max(a_sims))
            a_w = np.exp((a_sims - max_direct_sim) / self.temp)
            a_w /= (float(np.sum(a_w)) + EPS)
            a_attr = a_w @ A_mat
            a_norm = float(np.sqrt(np.dot(a_attr, a_attr)))
            if a_norm > EPS:
                a_attr /= a_norm
            lip1 = 1.0 / PHI
            x_a = (1.0 - lip1) * x0 + lip1 * a_attr
            xa_norm = float(np.sqrt(np.dot(x_a, x_a)))
            if xa_norm > EPS:
                x_a /= xa_norm
            fp_sim = 0.70 * float(np.dot(x0, x_a)) + 0.30 * float(np.dot(x0, xt))
        else:
            max_direct_sim = float(np.max(C @ x0))
            fp_sim = float(np.dot(x0, xt))

        # Length-asymmetry containment adjustment for terse queries (2 <= n_words <= 7)
        n_words = len(_WORD_RE.findall(prompt.lower()))
        if 2 <= n_words <= 7 and max_direct_sim >= 0.06:
            scale = m.sqrt(min(2.2, 9.0 / max(2, n_words)))
            max_direct_sim = min(1.0, max_direct_sim * scale)

        resonance = max(0.0, min(1.0, 0.5 * max_direct_sim + 0.5 * fp_sim))
        tot_diff = xt - x0
        total_disp = float(np.sqrt(np.dot(tot_diff, tot_diff)))

        if mutate:
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
