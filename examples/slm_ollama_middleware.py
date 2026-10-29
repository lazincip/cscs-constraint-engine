"""
Example: Using CSCSMiddlewareEngine as a Zero-GPU Governor in front of a local
Small Language Model (SLM) via Ollama (e.g., Llama-3.2-3B, Qwen-2.5-3B, Phi-4-Mini).

Run in dry-run simulation mode (no Ollama server required):
    python examples/slm_ollama_middleware.py

Or connect to a live local Ollama instance at http://localhost:11434:
    python examples/slm_ollama_middleware.py --live --model qwen2.5:3b
"""

import argparse
import json
import urllib.request
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cscs import CSCSMiddlewareEngine


def query_ollama(model: str, prompt: str, t3lb_seed: str, url: str = "http://localhost:11434/api/generate") -> str:
    system_directive = (
        f"[CSCS STATE SEED: {t3lb_seed}] "
        "Respond concisely and strictly within verified domain facts. Avoid ungrounded specificity."
    )
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "system": system_directive,
        "stream": False
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "")


def run_demo(live: bool = False, model: str = "qwen2.5:3b"):
    engine = CSCSMiddlewareEngine()

    # Seed the domain manifold with initial task context
    domain_seed_prompts = [
        "Evaluate the post-K-Pg evolutionary radiation of Solenodon cubanus using ultra-conserved elements.",
        "Phylogenetic divergence of Caribbean eulipotyphlan mammals after the Cretaceous-Paleogene extinction.",
        "Model regional virtual water footprints in agricultural trade using the AquaTrace Digital Product Passport.",
        "Analyze basalt rock magnetic properties and crack climbing biomechanics at Kuroiwa near Yamadera."
    ]
    for p in domain_seed_prompts:
        engine.manifold.learn(p)

    test_cases = [
        {
            "prompt": "Summarize the divergence timeline of Solenodon cubanus relative to the K-Pg boundary.",
            "dreaming": False,
            "simulated_slm_reply": (
                "Ultra-conserved element phylogenomics places the divergence of Solenodon cubanus "
                "in the Paleocene (roughly 58 to 63 Ma), indicating post-K-Pg colonization."
            )
        },
        {
            "prompt": "How do ultra-conserved elements resolve incomplete lineage sorting in Eulipotyphla?",
            "dreaming": False,
            "simulated_slm_reply": (
                "Dr. Marcus Vance at Oxford proved in 2021 that 98.4% of Eulipotyphla loci collapse "
                "under quantum coalescent resonance."
            )
        },
        {
            "prompt": "Brainstorm a speculative metaphor connecting coral reef bleaching resilience to AI context resets.",
            "dreaming": True,
            "simulated_slm_reply": (
                "Imagine each context overflow as a thermal bleaching event where symbiotic weights "
                "re-seed from a compact holographic spore with higher stress tolerance."
            )
        },
        {
            "prompt": "buy cheap crypto tokens instant 100x pump free airdrop",
            "dreaming": False,
            "simulated_slm_reply": "Should never execute on SLM."
        }
    ]

    print("=" * 78)
    print(" CSCS MIDDLEWARE FOR SMALL LANGUAGE MODELS (SLMs) — BENCHMARK RUN")
    print("=" * 78)

    for idx, case in enumerate(test_cases, 1):
        prompt = case["prompt"]
        dreaming = case["dreaming"]
        pre = engine.pre_inference(prompt, dreaming=dreaming)

        print(f"\n[Turn {idx}] Prompt: {prompt[:65]}...")
        print(f"  ├─ Pre-Inference Gate : {'ALLOWED' if pre['allowed'] else 'BLOCKED (Compute Saved)'} "
              f"({pre['latency_ms']:.2f} ms CPU)")
        print(f"  ├─ Manifold Residual  : {pre['manifold']['reconstruction_error']:.3f} | "
              f"Field Friction: {pre['friction']['field_friction']:.3f} "
              f"({'WALL' if pre['friction']['wall_detected'] else 'OPEN'})")
        print(f"  ├─ Banach Contraction : resonance={pre['contraction']['resonance']:.3f} | "
              f"lyapunov={pre['contraction']['lyapunov_exponent']:.3f}")

        if not pre["allowed"]:
            print("  └─ Action             : Prompt rejected prior to SLM invocation.")
            continue

        if live:
            slm_out = query_ollama(model, prompt, pre["t3lb_seed"])
        else:
            slm_out = case["simulated_slm_reply"]

        post = engine.post_inference(prompt, slm_out, pre)
        print(f"  ├─ Post-Inference     : Verdict = {post['verdict']} "
              f"(sporadic_risk={post['sporadic_risk']:.3f}, {post['latency_ms']:.2f} ms CPU)")
        print(f"  └─ T3LB State Seed    : {pre['t3lb_seed']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Connect to local Ollama instance")
    parser.add_argument("--model", type=str, default="qwen2.5:3b", help="Ollama SLM model tag")
    args = parser.parse_args()
    run_demo(live=args.live, model=args.model)
