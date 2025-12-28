# ============================================================
# SEER-NLP : Pension & Disaster Assistance Decision Portal
# FINAL GOVERNMENT-GRADE SUBMISSION VERSION
# ============================================================

import streamlit as st
import pickle
import json
import pandas as pd
from datetime import datetime
import os
from typing import List

# ============================================================
# REQUIRED FOR PICKLE DESERIALIZATION
# ============================================================
import seer_model  # noqa: F401

# ============================================================
# CONFIGURATION
# ============================================================

ARTIFACT_DIR = "seer_artifacts"

st.set_page_config(
    page_title="SEER-NLP | Pension & Disaster Assistance",
    page_icon="🏛️",
    layout="wide"
)

# ============================================================
# GOVERNMENT UI THEME
# ============================================================

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Segoe UI", Roboto, Arial, sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
}

section[data-testid="stSidebar"] {
    background-color: #020617;
    border-right: 1px solid #1e293b;
}

h1, h2, h3 {
    color: #f8fafc;
    font-weight: 600;
}

.card {
    background-color: #020617;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
}

textarea {
    background-color: #020617 !important;
    border: 1px solid #334155 !important;
    color: #f8fafc !important;
    border-radius: 12px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    color: white;
    border-radius: 12px;
    padding: 0.6em 1.6em;
    border: none;
    font-weight: 600;
}

[data-testid="metric-container"] {
    background-color: #020617;
    border: 1px solid #1e293b;
    padding: 18px;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD PIPELINE ARTIFACTS
# ============================================================

@st.cache_resource(show_spinner="🔄 Loading SEER-NLP engines…")
def load_artifacts():
    with open(os.path.join(ARTIFACT_DIR, "stage1_normalizer.pkl"), "rb") as f:
        normalizer = pickle.load(f)

    with open(os.path.join(ARTIFACT_DIR, "stage2_semantic_engine.pkl"), "rb") as f:
        semantic_engine = pickle.load(f)

    with open(os.path.join(ARTIFACT_DIR, "stage3_decision_engine.pkl"), "rb") as f:
        decision_engine = pickle.load(f)

    return normalizer, semantic_engine, decision_engine


normalizer, semantic_engine, decision_engine = load_artifacts()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🏛️ SEER-NLP")
    st.caption("Decision Support System")
    st.divider()

    mode = st.radio(
        "Operational Mode",
        [
            "📝 Single Application Review",
            "📂 Batch Screening",
            "🧪 Policy Scenario Simulation"
        ]
    )

# ============================================================
# HEADER
# ============================================================

st.title("📄 Pension & Disaster Relief Decision Portal")
st.caption("Deterministic • Explainable • Rule-Based • Officer-Assistive")

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def extract_disaster_types(facts: List[str]) -> List[str]:
    disasters = {
        "flood": "Flood",
        "earthquake": "Earthquake",
        "cyclone": "Cyclone",
        "fire": "Fire",
        "landslide": "Landslide",
        "drought": "Drought",
        "storm": "Storm"
    }
    found = []
    for f in facts:
        for k, v in disasters.items():
            if k in f.lower() and v not in found:
                found.append(v)
    return found


def detect_text_column(df: pd.DataFrame):
    keys = ["text", "narrative", "description", "remarks", "details", "statement"]
    for c in df.columns:
        if any(k in c.lower() for k in keys):
            return c
    return None


def run_pipeline(raw_text: str) -> dict:
    s1 = normalizer.normalize(raw_text)
    s2 = semantic_engine.analyze(s1["normalized_text"])
    s3 = decision_engine.decide(s2)

    # FINAL VERDICT SANITIZATION
    if "ELIGIBLE" in s3["Final_Verdict"].upper():
        s3["Final_Verdict"] = "ELIGIBLE"
    else:
        s3["Final_Verdict"] = "NOT ELIGIBLE"

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "analysis": s2,
        "decision": s3,
        "noise": s1["noise_score"]
    }


def render_vulnerability(v):
    df = pd.DataFrame(v.items(), columns=["Dimension", "Score"])
    df.set_index("Dimension", inplace=True)
    st.bar_chart(df)

# ============================================================
# MODE 1 — SINGLE APPLICATION
# ============================================================

if mode == "📝 Single Application Review":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Applicant Narrative")

    text = st.text_area(
        "Enter applicant’s written statement",
        height=220,
        placeholder=(
            "Example:\n"
            "I am 66 years old. Flood destroyed my house. "
            "I am a widow and have no income."
        )
    )

    analyze = st.button("Evaluate Application", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if analyze and text.strip():

        result = run_pipeline(text)
        a = result["analysis"]
        d = result["decision"]

        # ================= DECISION SUMMARY =================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Decision Summary")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Verdict", d["Final_Verdict"])
        c2.metric("Priority Level", d["Priority_Level"])
        c3.metric("Age Group", a["Age_Group"])
        c4.metric("Disaster Severity", a["Disaster_Severity"])
        st.markdown('</div>', unsafe_allow_html=True)

        # ================= KEY EXTRACTED DETAILS =================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Key Extracted Applicant Details")

        r1c1, r1c2, r1c3, r1c4 = st.columns(4)

        r1c1.markdown("**Applicant Age**")
        r1c1.markdown(f"### {a.get('Age', 'Not specified')}")

        r1c2.markdown("**Age Category**")
        r1c2.markdown(f"### {a['Age_Group']}")

        disaster_types = extract_disaster_types(a["Extracted_Facts"])
        r1c3.markdown("**Disaster Type(s)**")
        r1c3.markdown(f"### {', '.join(disaster_types) if disaster_types else 'Not stated'}")

        r1c4.markdown("**Disaster Impact**")
        r1c4.markdown(f"### {a['Disaster_Severity']}")

        st.divider()

        st.markdown("### Socio-Economic & Health Indicators")

        flags = {
            "widow": "Widow status identified",
            "no income": "No income / livelihood loss",
            "disability": "Disability related vulnerability",
            "old age": "Old-age dependency",
            "no shelter": "Housing loss reported"
        }

        shown = False
        cols = st.columns(3)
        idx = 0

        for k, label in flags.items():
            if k in a["Extracted_Facts"]:
                cols[idx % 3].success(label)
                idx += 1
                shown = True

        if not shown:
            st.info("No explicit vulnerability indicators detected from narrative.")

        st.divider()

        st.markdown("### Administrative Interpretation")

        if a["Age_Group"] == "Senior":
            st.markdown("• Applicant qualifies under senior-citizen criteria.")
        if disaster_types:
            st.markdown("• Disaster impact reported affecting living conditions.")
        if "no income" in a["Extracted_Facts"]:
            st.markdown("• Economic distress due to absence of income.")
        if "widow" in a["Extracted_Facts"]:
            st.markdown("• Widow status increases social dependency.")

        st.markdown('</div>', unsafe_allow_html=True)

        # ================= PENSION =================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Pension & Assistance Determination")

        p = d["Pension"]
        st.markdown(f"**Processing State:** {p['State']}")
        st.markdown(f"**Primary Pension:** {p['Primary'] or '—'}")

        if p["Types"]:
            for t in p["Types"]:
                st.success(t)
        else:
            st.warning("No pension eligibility identified.")

        st.markdown('</div>', unsafe_allow_html=True)

        # ================= VULNERABILITY =================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Vulnerability Profile")
        render_vulnerability(a["Vulnerability_Profile"])
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# MODE 2 — BATCH SCREENING (ELIGIBILITY ONLY)
# ============================================================

elif mode == "📂 Batch Screening":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Batch Screening (Eligibility Only)")
    file = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    if file:
        df = pd.read_csv(file)
        col = detect_text_column(df)

        if not col:
            st.error("No narrative column found.")
        else:
            if st.button("Process Batch", type="primary"):
                outputs = [run_pipeline(t) for t in df[col].dropna().astype(str)]

                summary = pd.DataFrame([
                    {
                        "Final Verdict": o["decision"]["Final_Verdict"],
                        "Age Group": o["analysis"]["Age_Group"],
                        "Disaster Severity": o["analysis"]["Disaster_Severity"],
                        "Primary Pension": o["decision"]["Pension"]["Primary"] or "None"
                    }
                    for o in outputs
                ])

                st.dataframe(summary, use_container_width=True)

# ============================================================
# MODE 3 — POLICY SIMULATION
# ============================================================

else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Policy Scenario Simulation")

    age = st.slider("Age", 18, 100, 65)
    disaster = st.selectbox("Disaster Event", ["None", "Flood", "Earthquake", "Cyclone", "Fire"])
    widow = st.checkbox("Widow")
    no_income = st.checkbox("No Income")

    text = f"I am {age} years old. "
    if disaster != "None":
        text += f"I was affected by {disaster.lower()}. "
    if widow:
        text += "I am a widow. "
    if no_income:
        text += "I have no income. "

    st.code(text)

    if st.button("Run Simulation"):
        st.json(run_pipeline(text)["decision"])
