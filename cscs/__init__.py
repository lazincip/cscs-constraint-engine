"""
CSCS Constraint Physics Engine (v2026.10.1)
Principal Investigator & Author: Lazaro M. Echenique-Diaz, Ph.D.
"""

from .manifold import SpectralManifold
from .contraction import BanachContractionMirror
from .scout import RBFConstraintScout
from .compression import MDLCompressionRouter
from .verifier import DriftVerifier
from .codec import T3LBStateCodec
from .engine import CSCSMiddlewareEngine

__all__ = [
    "SpectralManifold",
    "BanachContractionMirror",
    "RBFConstraintScout",
    "MDLCompressionRouter",
    "DriftVerifier",
    "T3LBStateCodec",
    "CSCSMiddlewareEngine"
]
__version__ = "2026.10.1"
