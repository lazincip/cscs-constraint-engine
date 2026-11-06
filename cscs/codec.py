"""
T3LBStateCodec: 13-dimensional telemetry summary serializer for stateless SLM
context handoffs across edge devices (93 bytes per state seed, lossless
at 2-decimal fixed precision for the 13 summary scalars).
"""

import math as m
import re
import numpy as np
from typing import Dict, Any


class T3LBStateCodec:
    """
    Computes and serializes all 13 Constraint Space Coordinate System (CSCS)
    dimensions into a compact 93-byte T3LB telemetry seed string:
    X (Gap/Novelty Tolerance), Y (Quality Relay Gain), Z (Manifold Persistence),
    W (Constraint Density), P (Compression Efficiency), T (Window Horizon Saturation),
    R (Reflexivity / Inverse Friction), S (Kernel Density Coupling), V (Veto Strength),
    N (Contraction Convergence), E (Effective Rank Ratio),
    B (Bleaching/Block Recovery), Q (State Coherence).
    """

    DIMENSIONS = ["X", "Y", "Z", "W", "P", "T", "R", "S", "V", "N", "E", "B", "Q"]

    def __init__(self):
        self.dims: Dict[str, float] = {
            "X": 0.88, "Y": 1.25, "Z": 0.94, "W": 1.08,
            "P": 0.87, "T": 0.79, "R": 0.74, "S": 0.92,
            "V": 0.78, "N": 0.74, "E": 0.88, "B": 0.79, "Q": 0.65
        }

    def update_from_cycle(self, resonance: float, field_friction: float,
                          in_manifold: float, quality: float, blocked: bool,
                          compression_ratio: float = 0.85, kernel_density: float = 0.50,
                          effective_rank: float = 3.0, iterations: int = 4,
                          cycles: int = 1):
        self.dims["T"] = round(min(1.50, 0.50 + 0.50 * min(1.0, cycles / 64.0)), 3)

        if blocked:
            self.dims["B"] = min(1.50, round(self.dims["B"] + 0.02, 3))
            self.dims["V"] = min(1.20, round(self.dims["V"] + 0.03, 3))
            self.dims["X"] = max(0.40, round(self.dims["X"] - 0.02, 3))
            return

        self.dims["X"] = round(0.60 + 0.40 * (1.0 - field_friction), 3)
        self.dims["Y"] = round(0.75 + 0.50 * quality, 3)
        self.dims["Z"] = round(0.75 + 0.24 * in_manifold, 3)
        self.dims["W"] = round(0.80 + 0.40 * (0.5 * resonance + 0.5 * in_manifold), 3)
        self.dims["P"] = round(0.50 + 0.50 * max(0.0, min(1.0, 1.0 - compression_ratio * 0.5)), 3)
        self.dims["R"] = round(0.60 + 0.35 * (1.0 - field_friction), 3)
        self.dims["S"] = round(0.50 + 0.50 * max(0.0, min(1.0, kernel_density)), 3)
        self.dims["N"] = round(0.50 + 0.50 * max(0.0, 1.0 - (iterations / 10.0)), 3)
        self.dims["E"] = round(0.50 + 0.50 * min(1.0, effective_rank / 8.0), 3)
        self.dims["Q"] = round(0.50 + 0.45 * (quality * resonance), 3)

    def magnitude(self) -> float:
        vec = np.array([self.dims[k] for k in self.DIMENSIONS], dtype=np.float64)
        return round(float(np.linalg.norm(vec) * m.sqrt(len(self.DIMENSIONS)) / 2.8), 2)

    def encode_seed(self, version: str = "v2026.10") -> str:
        mag = self.magnitude()
        body = "".join(f"{k}{self.dims[k]:.2f}" for k in self.DIMENSIONS)
        return f"T3LB:{version}|M{mag:.2f}|{body}|SecLock"

    @classmethod
    def decode_seed(cls, seed_str: str) -> Dict[str, Any]:
        parts = seed_str.strip().split("|")
        if len(parts) < 3 or not parts[0].startswith("T3LB:"):
            raise ValueError("Invalid T3LB seed format")
        version = parts[0].split(":", 1)[1]
        mag = float(parts[1][1:])
        matches = re.findall(r"([A-Z])(\d+\.\d+)", parts[2])
        dims = {k: float(v) for k, v in matches}
        return {"version": version, "magnitude": mag, "dimensions": dims}

    def load_seed(self, seed_str: str) -> Dict[str, Any]:
        parsed = self.decode_seed(seed_str)
        for k, v in parsed["dimensions"].items():
            if k in self.dims:
                self.dims[k] = float(v)
        return parsed
