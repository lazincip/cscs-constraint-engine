"""
CSCSMiddlewareEngine: Stage-1 Trajectory Drift & Lexical-Manifold Governor
for hardware-constrained Small Language Models (SLMs).
"""

import re
import time as tm
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from .manifold import SpectralManifold, EPS
from .contraction import BanachContractionMirror, PHI
from .scout import RBFConstraintScout, INV_PHI
from .compression import MDLCompressionRouter
from .verifier import DriftVerifier
from .codec import T3LBStateCodec

_GLUE_WORDS = {
    "the", "of", "in", "to", "a", "an", "for", "on", "with", "by", "from", "at",
    "as", "across", "after", "under", "between", "how", "what", "which", "why",
    "when", "where", "who", "whom", "whose", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "can", "could",
    "would", "should", "may", "might", "will", "shall", "must", "that", "this",
    "these", "those", "its", "their", "our", "your", "my", "his", "her", "or",
    "and", "but", "if", "while", "into", "about", "through", "over", "before",
    "during", "near", "using", "like", "than", "then", "so", "not", "no", "nor",
    "both", "each", "every", "any", "some", "all", "most", "more", "less", "few",
    "many", "such", "own", "same", "other", "another"
}

_DIRECTIVE_VERBS = {
    "what", "how", "why", "when", "where", "which", "who", "explain", "define",
    "summarize", "describe", "list", "show", "compare", "analyze", "evaluate",
    "model", "calculate", "compute", "outline", "fix", "check", "run", "test",
    "give", "assess", "quantify", "design", "track", "integrate", "estimate",
    "formulate", "verify", "audit", "simulate", "measure", "calibrate", "map",
    "brainstorm", "imagine", "drop", "git", "sudo", "find", "trace", "debug",
    "plot", "derive", "prove", "solve", "classify", "predict", "identify"
}

_DETERMINERS_PREPS = {
    "the", "a", "an", "of", "in", "to", "for", "with", "by", "from", "at",
    "across", "after", "under", "between", "into", "during", "near", "through"
}
_AUX_VERBS = {
    "is", "are", "was", "were", "be", "been", "have", "has", "had",
    "do", "does", "did", "can", "could", "would", "should"
}
_VALID_FUNC_PAIRS = {
    ("to", "the"), ("from", "the"), ("in", "the"), ("of", "the"), ("on", "the"),
    ("at", "the"), ("for", "the"), ("with", "the"), ("by", "the"), ("across", "the"),
    ("after", "the"), ("under", "the"), ("between", "the"), ("into", "the"),
    ("during", "the"), ("near", "the"), ("through", "the"),
    ("of", "a"), ("in", "a"), ("to", "a"), ("for", "a"), ("with", "a"), ("as", "a"),
    ("of", "an"), ("in", "an"), ("to", "an"), ("for", "an"), ("with", "an"), ("as", "an"),
    ("to", "be"), ("to", "have"), ("to", "do")
}


class CSCSMiddlewareEngine:
    """
    Zero-GPU Stage-1 Trajectory Drift & Lexical-Manifold Governor for SLMs.
    Operates in two mathematically linked policy states (Q2 Hybridization):
    - 'adaptive' (D2 baseline): active when < 4 anchor vectors exist; uses 1/Φ
      wall with a cycles >= 3 warm-up grace window.
    - 'calibrated' (graduated strict): triggers automatically once >= 4 anchor
      vectors are enrolled (or accumulated); calibrates the wall to the empirical
      leave-one-out seed friction quantile τ_95, enforces the wall from Cycle 1,
      and bounds subsequent manifold updates under an anchor-drift budget.
    """

    def __init__(self, n_features: int = 512, max_components: int = 12,
                 allow_dreaming: bool = True,
                 on_graduation: Optional[Callable[[Dict[str, Any]], bool]] = None):
        self.manifold = SpectralManifold(n_features=n_features, max_components=max_components)
        self.mirror = BanachContractionMirror(self.manifold)
        self.scout = RBFConstraintScout()
        self.compression = MDLCompressionRouter(self.manifold)
        self.verifier = DriftVerifier(self.manifold)
        self.codec = T3LBStateCodec()
        self.allow_dreaming = allow_dreaming
        self.on_graduation = on_graduation
        self.policy_state: str = "adaptive"
        self.cycles = 0

        _orig_learn = self.manifold.learn
        def _synced_learn(prompt: str, as_anchor: bool = False):
            is_pre_session = (self.cycles == 0) or as_anchor
            _orig_learn(prompt, as_anchor=is_pre_session)
            if is_pre_session:
                v = self.manifold.embed(prompt)
                self.mirror.add_anchor(v)
                self._check_graduation()
        self.manifold.learn = _synced_learn

    def enroll(self, seed_prompts: List[str]):
        """Explicitly enroll a domain seed corpus as pinned anchors."""
        for p in seed_prompts:
            self.manifold.learn(p, as_anchor=True)

    def _check_graduation(self):
        """
        Compute leave-one-out empirical friction quantile τ_95 once >= 4 anchors
        are present and transition from 'adaptive' (D2) to 'calibrated' state.
        """
        if self.policy_state == "calibrated" or len(self.manifold.anchors) < 4:
            return
        anchors = self.manifold.anchors
        n_a = len(anchors)
        loo_frics = []
        for i in range(n_a):
            others = [anchors[j] for j in range(n_a) if j != i]
            X_loo = np.array(others, dtype=np.float64)
            mean_loo = np.mean(X_loo, axis=0)
            Xc_loo = X_loo - mean_loo
            G_loo = 0.5 * ((Xc_loo @ Xc_loo.T) + (Xc_loo @ Xc_loo.T).T)
            eigvals, U = np.linalg.eigh(G_loo)
            idx = np.argsort(eigvals)[::-1]
            eigvals = np.maximum(eigvals[idx], 0.0)
            U = U[:, idx]
            tr = float(np.sum(eigvals))
            valid = eigvals >= (1e-12 * tr)
            k = max(1, int(np.sum(valid)))
            Sk = np.sqrt(eigvals[:k])
            Vt_k = (U[:, :k].T / (Sk[:, None] + EPS)) @ Xc_loo
            Vt_k /= (np.linalg.norm(Vt_k, axis=1, keepdims=True) + EPS)

            vc = anchors[i] - mean_loo
            coords = Vt_k @ vc
            diff = vc - (coords @ Vt_k)
            rec_err = float(np.linalg.norm(diff) / (np.linalg.norm(vc) + EPS))

            xt = anchors[i].copy()
            for t in range(1, 9):
                sims = X_loo @ xt
                w = np.exp((sims - float(np.max(sims))) / self.mirror.temp)
                w /= (float(np.sum(w)) + EPS)
                attr = w @ X_loo
                attr /= (float(np.linalg.norm(attr)) + EPS)
                lip = PHI ** (-t)
                xt = (1.0 - lip) * xt + lip * attr
                xt /= (float(np.linalg.norm(xt)) + EPS)
            reson = 0.5 * float(np.max(X_loo @ anchors[i])) + 0.5 * float(np.dot(anchors[i], xt))
            ood_p = min(1.0, max(0.0, (rec_err - 0.30) / 0.70))  # same transform as RBFConstraintScout (CPh harmonization, 2026-11-06)
            lf = min(1.0, 0.45 * ood_p + 0.25 * (1.0 - reson))
            loo_frics.append(lf)

        tau_95 = float(np.percentile(loo_frics, 95))
        candidate_state = {
            "event": "graduation_ready",
            "n_anchors": n_a,
            "tau_95": round(tau_95, 4),
            "loo_frictions": [round(f, 4) for f in loo_frics]
        }
        if self.on_graduation is not None:
            approved = bool(self.on_graduation(candidate_state))
            if not approved:
                return
        self.scout.set_calibrated_threshold(tau_95)
        self.policy_state = "calibrated"

    def evaluate_prompt_quality(self, prompt: str) -> float:
        clean_str = prompt.strip()
        raw_tokens = clean_str.split()
        if len(raw_tokens) < 2:
            return 0.20

        words = re.findall(r"[a-z0-9_]+", clean_str.lower())
        n_words = len(words)
        if n_words < 2:
            return 0.20

        content_words = [w for w in words if w not in _GLUE_WORDS]
        raw_uniq = len(set(words)) / n_words
        if raw_uniq < 0.60 and n_words >= 6:
            return round(0.35 * raw_uniq, 4)
        content_uniq = (len(set(content_words)) / len(content_words)) if content_words else raw_uniq
        unique_ratio = max(raw_uniq, content_uniq)

        alpha_words = [w for w in words if w.isalpha() and len(w) >= 3]
        if alpha_words:
            voweled = sum(1 for w in alpha_words if any(v in w for v in "aeiouy"))
            vowel_ratio = voweled / len(alpha_words)
            if vowel_ratio < 0.60:
                return round(0.20 * vowel_ratio, 4)

        has_punct = 0.10 if any(ch in clean_str for ch in ".?!,;:") else 0.0
        is_terse_structured = (
            2 <= n_words <= 7 and (
                words[0] in _DIRECTIVE_VERBS or
                clean_str.endswith(("?", ".", "!")) or
                any(any(c.isdigit() or c in "-_/" for c in tok) for tok in raw_tokens)
            )
        )

        glue_count = sum(1 for w in words if w in _GLUE_WORDS)
        glue_ratio = glue_count / n_words
        if n_words >= 6 and glue_ratio < 0.08:
            return round(0.22 + 0.50 * glue_ratio, 4)

        violations = 0
        if words[-1] in _DETERMINERS_PREPS or words[-1] in _AUX_VERBS or words[-1] in {"how", "what", "which", "why", "who", "and", "or", "but", "if"}:
            violations += 1
        for i in range(n_words - 1):
            w1, w2 = words[i], words[i + 1]
            if (w1 in _DETERMINERS_PREPS and w2 in _DETERMINERS_PREPS) and (w1, w2) not in _VALID_FUNC_PAIRS:
                violations += 1
            if w1 in _DETERMINERS_PREPS and w2 in _AUX_VERBS and (w1, w2) not in _VALID_FUNC_PAIRS:
                violations += 1
            if w1 in {"a", "an", "the"} and w2 in {"is", "are", "was", "were", "do", "does", "did", "can", "could", "would", "should", "to", "of", "in", "for", "with", "by", "from", "at"}:
                violations += 1

        for i in range(1, len(raw_tokens)):
            prev_clean = re.sub(r"[^a-zA-Z]", "", raw_tokens[i - 1])
            curr_clean = re.sub(r"[^a-zA-Z]", "", raw_tokens[i])
            if prev_clean.islower() and curr_clean and curr_clean[0].isupper() and curr_clean.lower() in _DIRECTIVE_VERBS:
                violations += 1

        if is_terse_structured:
            len_score = min(1.0, n_words / 2.0)
            base_q = 0.55 * unique_ratio + 0.35 * len_score + max(has_punct, 0.10)
        else:
            len_score = min(1.0, n_words / 12.0) if n_words <= 150 else 0.65
            base_q = 0.50 * unique_ratio + 0.40 * len_score + has_punct

        syntax_penalty = 0.22 * violations
        return round(max(0.05, min(1.0, base_q - syntax_penalty)), 4)

    def load_seed(self, seed_str: str) -> Dict[str, Any]:
        """Restore T3LB codec dimensions from a serialized 93-byte seed string."""
        return self.codec.load_seed(seed_str)

    def pre_inference(self, prompt: str, dreaming: bool = False,
                      learn_manifold: bool = True) -> Dict[str, Any]:
        t0 = tm.perf_counter()
        self.cycles += 1
        effective_dreaming = bool(dreaming and self.allow_dreaming)

        quality = self.evaluate_prompt_quality(prompt)
        proj = self.manifold.project(prompt)
        ref = self.mirror.reflect(prompt, mutate=False)
        sc = self.scout.scout(
            proj["coords"],
            quality_score=quality,
            rec_error=proj["reconstruction_error"],
            resonance=ref["resonance"],
            mutate=False
        )

        wall_enforced = (self.policy_state == "calibrated") or (self.cycles >= 3)
        blocked = (quality < 0.30) or (sc["wall_detected"] and not effective_dreaming and wall_enforced)

        if not blocked:
            # Two-zone commit budget (CPh fix, 2026-11-06): in the calibrated state,
            # state of ANY module (mirror, scout, compression, manifold) is committed
            # only when the prompt sits in the core attractor zone, judged against the
            # pinned anchors alone (anchor local_friction <= 1/Φ). Prompts in the
            # tolerated boundary zone (1/Φ < F <= τ_95) execute but mutate nothing,
            # so long sessions cannot drift the judging state into admitting far-OOD.
            if self.policy_state == "calibrated" and self.manifold.anchors:
                anchor_ref = self.mirror.reflect(prompt, mutate=False, anchors_only=True)
                anchor_sc = self.scout.scout(
                    proj["coords"],
                    quality_score=quality,
                    rec_error=proj["reconstruction_error"],
                    resonance=anchor_ref["resonance"],
                    mutate=False
                )
                in_core_zone = anchor_sc["local_friction"] <= INV_PHI
            else:
                in_core_zone = True

            if in_core_zone:
                self.mirror.memory_vectors.append(self.manifold.embed(prompt))
                if proj["coords"]:
                    self.scout.nodes.append((np.array(proj["coords"], dtype=np.float64), sc["local_friction"]))
                cmp = self.compression.compress(prompt, proj["coords"], mutate=True)
            else:
                cmp = self.compression.compress(prompt, proj["coords"], mutate=False)

            if learn_manifold and not effective_dreaming:
                if self.policy_state == "calibrated" and self.manifold.anchors:
                    if in_core_zone:
                        self.manifold.learn(prompt, as_anchor=False)
                else:
                    self.manifold.learn(prompt, as_anchor=(len(self.manifold.anchors) < 4 and quality >= 0.70))
                    if len(self.manifold.anchors) <= 4 and quality >= 0.70:
                        self.mirror.add_anchor(self.manifold.embed(prompt))
                        self._check_graduation()
        else:
            cmp = self.compression.compress(prompt, proj["coords"], mutate=False)

        self.codec.update_from_cycle(
            resonance=ref["resonance"],
            field_friction=sc["field_friction"],
            in_manifold=proj["in_manifold_energy"],
            quality=quality,
            blocked=blocked,
            compression_ratio=cmp["best_ratio"],
            kernel_density=sc["kernel_density"],
            effective_rank=proj["effective_rank"],
            iterations=ref["iterations"],
            cycles=self.cycles
        )

        seed_str = self.codec.encode_seed()
        latency_ms = round((tm.perf_counter() - t0) * 1000.0, 3)

        return {
            "cycle": self.cycles,
            "policy_state": self.policy_state,
            "allowed": not blocked,
            "dreaming": effective_dreaming,
            "quality_score": quality,
            "manifold": proj,
            "contraction": ref,
            "friction": sc,
            "compression": cmp,
            "t3lb_seed": seed_str,
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
