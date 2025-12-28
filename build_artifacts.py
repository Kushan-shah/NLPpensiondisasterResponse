# ============================================================
# SEER-NLP : STAGE 4 — MULTI-PICKLE ARTIFACT BUILDER (FINAL)
# ============================================================
# Purpose:
# - Persist each stage independently
# - Avoid __main__ pickle errors
# - Enable modular loading in Streamlit
#
# RUN THIS FILE ONCE
# ============================================================

import os
import pickle
import hashlib
from datetime import datetime

# 🔥 CRITICAL: IMPORT FROM seer_model (NOT __main__)
from seer_model import (
    TextNormalizer,
    SemanticPolicyEngine,
    FinalDecisionEngine
)

# ============================================================
# CONFIG
# ============================================================

ARTIFACT_DIR = "seer_artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ============================================================
# INTERNAL SAVE HELPER
# ============================================================

def _save(obj, filename):
    path = os.path.join(ARTIFACT_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ Saved: {path}")


# ============================================================
# BUILD ALL ARTIFACTS
# ============================================================

def build_all_artifacts():
    print("\n🚀 Building SEER-NLP artifacts...\n")

    # -------------------------------
    # Stage 1 — Text Normalizer
    # -------------------------------
    normalizer = TextNormalizer()
    _save(normalizer, "stage1_normalizer.pkl")

    # -------------------------------
    # Stage 2 — Semantic Policy Engine
    # -------------------------------
    semantic_engine = SemanticPolicyEngine()
    _save(semantic_engine, "stage2_semantic_engine.pkl")

    # -------------------------------
    # Stage 3 — Final Decision Engine
    # -------------------------------
    decision_engine = FinalDecisionEngine()
    _save(decision_engine, "stage3_decision_engine.pkl")

    # -------------------------------
    # Metadata (Governance & Audit)
    # -------------------------------
    metadata = {
        "system": "SEER-NLP",
        "version": "v4.1",
        "architecture": "Multi-stage deterministic NLP",
        "stages": {
            "stage1": "TextNormalizer",
            "stage2": "SemanticPolicyEngine",
            "stage3": "FinalDecisionEngine"
        },
        "created_at": datetime.utcnow().isoformat(),
        "notes": [
            "Independent pickles for each stage",
            "Safe for Streamlit & API deployment",
            "No ML / Fully explainable",
            "Government-aligned pension reasoning"
        ]
    }

    fingerprint = hashlib.sha256(
        (metadata["system"] + metadata["version"]).encode()
    ).hexdigest()[:12]

    metadata["fingerprint"] = fingerprint

    _save(metadata, "metadata.pkl")

    print("\n🎉 ALL SEER-NLP ARTIFACTS GENERATED SUCCESSFULLY")
    print(f"🔐 Model Fingerprint: {fingerprint}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    build_all_artifacts()
