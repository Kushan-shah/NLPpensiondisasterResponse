# ============================================================
# SEER-NLP : REAL-WORLD TEXT NORMALIZATION ENGINE (v4.2 FINAL)
# ============================================================
# Purpose:
# - Accept noisy, informal, OCR / handwritten-like narratives
# - Normalize meaning → policy-aligned canonical facts
# - Guarantee compatibility with downstream semantic ontology
#
# Deterministic | Explainable | Zero ML | Production-safe
# ============================================================

import re
import unicodedata
from typing import Dict, List


class TextNormalizer:
    """
    Field-grade text normalization engine.

    Converts:
    - Broken English
    - Spoken / informal narratives
    - OCR / scanned text
    - Mixed Hindi-English phrases

    Into:
    - Canonical, ontology-aligned semantic text
    """

    def __init__(self):

        # ----------------------------------------------------
        # BASIC LEXICAL NORMALIZATION
        # ----------------------------------------------------
        self.basic_map = {
            # Age
            r"\byrs\b": "years",
            r"\byr\b": "year",
            r"\bage\s*[:=]\s*": "age ",
            r"\bmy age\b": "age",

            # Administration
            r"\bgovt\b": "government",
            r"\bdept\b": "department",
            r"\bdistt\b": "district",
            r"\baddr\b": "address",

            # Death / widowhood
            r"\bexpired\b": "passed away",
            r"\bdead\b": "passed away",
            r"\bhusband dead\b": "husband passed away",
            r"\blost my husband\b": "widow",
            r"\bhusband passed away\b": "widow",
            r"\bw/o\b": "widow of",

            # Income
            r"\bjobless\b": "no income",
            r"\bno work\b": "no income",
            r"\bdaily wage stopped\b": "no income",
            r"\bearning stopped\b": "no income",
            r"\bunemployed\b": "no income"
        }

        self.basic_patterns = [
            (re.compile(p), r) for p, r in self.basic_map.items()
        ]

        # ----------------------------------------------------
        # AGE EXPRESSIONS (ROBUST)
        # ----------------------------------------------------
        self.age_patterns = [
            re.compile(r"\bage\s*(\d{1,3})\b"),
            re.compile(r"\bmy age is\s*(\d{1,3})\b"),
            re.compile(r"\b(\d{1,3})\s*(yrs|years|yr|year)\b"),
            re.compile(r"\bi am\s*(\d{1,3})\b")
        ]

        # ----------------------------------------------------
        # HEALTH / DISABILITY
        # ----------------------------------------------------
        self.health_map = [
            (r"cant walk|cannot walk|leg problem", "physical disability"),
            (r"sick|ill|medical problem|health issue", "health condition"),
            (r"blind|cannot see", "visual disability"),
            (r"deaf|cannot hear", "hearing disability"),
            (r"bed ridden|bedridden", "severe disability"),
            (r"too old|old age problem", "old age")
        ]

        # ----------------------------------------------------
        # LIVELIHOOD & ASSET LOSS (ONTOLOGY-ALIGNED)
        # ----------------------------------------------------
        self.loss_map = [
            # Housing
            (r"house destroyed|home destroyed|house collapsed|roof fell", "housing loss"),
            (r"kacha house|pucca house|house broke|ghar gir gaya", "housing loss"),

            # Livelihood
            (r"shop closed|shop destroyed|business loss", "livelihood loss"),
            (r"lost my job|work stopped|no daily work", "livelihood loss"),

            # Agriculture
            (r"crop destroyed|field washed away|fasal nuksan", "agricultural loss"),

            # Livestock
            (r"cow died|buffalo died|animals died|cattle lost", "livestock loss"),

            # Documents
            (r"documents lost|papers lost|ration card lost|id lost", "document loss"),

            # Total loss (STRONG SIGNAL)
            (r"nothing left|lost everything|everything destroyed|sab kuch chala gaya",
             "total loss")
        ]

        # ----------------------------------------------------
        # DISASTER NORMALIZATION
        # ----------------------------------------------------
        self.disaster_map = [
            (r"flood|flooding|water entered|pani bhar gaya", "flood"),
            (r"cyclone|storm|andhi|tez hawa", "cyclone"),
            (r"fire|burnt|aag lag gayi|fire accident", "fire"),
            (r"earthquake|bhukamp|ground shaking", "earthquake")
        ]

        # ----------------------------------------------------
        # SOCIAL & POVERTY SIGNALS
        # ----------------------------------------------------
        self.social_map = [
            (r"no support|alone|no one to help", "social vulnerability"),
            (r"dependent on others|living on help", "dependency"),
            (r"please help|sir help|urgent help", "distress request")
        ]

        # ----------------------------------------------------
        # FORM / OCR CLEANING
        # ----------------------------------------------------
        self.line_breaks = re.compile(r"\n+")
        self.field_sep = re.compile(r"\s*[:\-]\s*")
        self.illegal_chars = re.compile(r"[^\w\s\.,;!?-]")
        self.multi_space = re.compile(r"\s+")

        # ----------------------------------------------------
        # NOISE INDICATORS
        # ----------------------------------------------------
        self.noise_patterns = [
            re.compile(p) for p in [
                r"\?\?\?",
                r"unknown",
                r"illegible",
                r"n/a",
                r"xxx",
                r"cannot read",
                r"blurred"
            ]
        ]

    # ========================================================
    # NORMALIZATION PIPELINE
    # ========================================================

    def normalize(self, text: str) -> Dict[str, object]:
        trace: List[str] = []
        noise_score = 0.0

        # 1. Unicode normalization
        text = unicodedata.normalize("NFKD", text)
        trace.append("unicode normalized")

        # 2. Lowercase
        text = text.lower()
        trace.append("lowercased")

        # 3. Line break normalization
        if "\n" in text:
            text = self.line_breaks.sub(". ", text)
            trace.append("joined broken lines")

        # 4. Field separator normalization
        if self.field_sep.search(text):
            text = self.field_sep.sub(" ", text)
            trace.append("normalized field separators")

        # 5. Remove OCR junk
        if self.illegal_chars.search(text):
            text = self.illegal_chars.sub(" ", text)
            trace.append("removed OCR noise")

        # 6. Canonical lexical replacements
        for p, r in self.basic_patterns:
            if p.search(text):
                text = p.sub(r, text)
                trace.append(f"canonicalized: {r}")

        # 7. Age normalization
        for p in self.age_patterns:
            if p.search(text):
                text = p.sub(r"\1 years", text)
                trace.append("normalized age")

        # 8. Health normalization
        for p, r in self.health_map:
            if re.search(p, text):
                text = re.sub(p, r, text)
                trace.append(f"normalized health → {r}")

        # 9. Disaster normalization
        for p, r in self.disaster_map:
            if re.search(p, text):
                text = re.sub(p, r, text)
                trace.append(f"normalized disaster → {r}")

        # 10. Loss normalization
        for p, r in self.loss_map:
            if re.search(p, text):
                text = re.sub(p, r, text)
                trace.append(f"normalized loss → {r}")

        # 11. Social normalization
        for p, r in self.social_map:
            if re.search(p, text):
                text = re.sub(p, r, text)
                trace.append(f"normalized social → {r}")

        # 12. Whitespace cleanup
        text = self.multi_space.sub(" ", text).strip()
        trace.append("normalized whitespace")

        # 13. Noise scoring
        for p in self.noise_patterns:
            if p.search(text):
                noise_score += 0.15

        return {
            "normalized_text": text,
            "normalization_trace": trace,
            "noise_score": round(min(noise_score, 1.0), 2)
        }
from typing import Set, Dict, List


# ============================================================
# 1. FACT EXTRACTION
# ============================================================

class FactExtractor:
    """
    Extracts ontology-aligned semantic facts
    from normalized text.
    """

    KNOWN_FACTS = {
        # Disaster
        "flood", "cyclone", "fire", "earthquake",

        # Loss
        "housing loss",
        "livelihood loss",
        "agricultural loss",
        "livestock loss",
        "document loss",
        "total loss",

        # Economic
        "no income",

        # Health
        "physical disability",
        "visual disability",
        "hearing disability",
        "severe disability",
        "old age",

        # Social
        "widow",
        "dependency",
        "social vulnerability",

        # Distress
        "distress request"
    }

    def extract(self, text: str) -> Set[str]:
        return {fact for fact in self.KNOWN_FACTS if fact in text}


# ============================================================
# 2. AGE & DEMOGRAPHIC ANALYSIS
# ============================================================

class AgeAnalyzer:
    """
    Extracts age and assigns demographic group.
    """

    def extract_age(self, text: str) -> int | None:
        tokens = text.split()
        for i, t in enumerate(tokens):
            if t.isdigit() and i + 1 < len(tokens) and tokens[i + 1] == "years":
                age = int(t)
                if 0 < age < 130:
                    return age
        return None

    def age_group(self, age: int | None) -> str:
        if age is None:
            return "Unknown"
        if age < 18:
            return "Minor"
        if age < 60:
            return "Adult"
        return "Senior"


# ============================================================
# 3. VULNERABILITY MODEL (CUMULATIVE)
# ============================================================

class VulnerabilityModel:
    """
    Computes socio-economic vulnerability using
    cumulative, capped scoring.
    """

    def compute(self, age_group: str, facts: Set[str]) -> Dict[str, float]:
        economic = 0.0
        health = 0.0
        social = 0.0

        # ---------- Economic ----------
        if "no income" in facts:
            economic += 0.4
        if "livelihood loss" in facts or "agricultural loss" in facts:
            economic += 0.3
        if "total loss" in facts:
            economic += 0.5

        # ---------- Health ----------
        if "severe disability" in facts:
            health += 0.8
        elif "physical disability" in facts:
            health += 0.5
        if age_group == "Senior" or "old age" in facts:
            health += 0.3

        # ---------- Social ----------
        if "widow" in facts:
            social += 0.4
        if "dependency" in facts:
            social += 0.4
        if "social vulnerability" in facts:
            social += 0.3

        return {
            "economic": round(min(economic, 1.0), 2),
            "health": round(min(health, 1.0), 2),
            "social": round(min(social, 1.0), 2)
        }


# ============================================================
# 4. DISASTER ONTOLOGY (REALISTIC SEVERITY)
# ============================================================

class DisasterOntology:
    """
    Determines disaster severity using
    cumulative loss & impact scoring.
    """

    def severity(self, facts: Set[str]) -> str:
        score = 0

        # Disaster presence
        if any(d in facts for d in ["flood", "cyclone", "fire", "earthquake"]):
            score += 1

        # Loss escalation
        if "housing loss" in facts:
            score += 2
        if "livelihood loss" in facts:
            score += 2
        if "agricultural loss" in facts:
            score += 1
        if "livestock loss" in facts:
            score += 1
        if "document loss" in facts:
            score += 1
        if "total loss" in facts:
            score += 3

        if score >= 6:
            return "Severe"
        elif score >= 3:
            return "Moderate"
        elif score >= 1:
            return "Low"
        return "None"

    GOVERNMENT_ACTIONS = {
        "None": [
            "No disaster response required"
        ],
        "Low": [
            "Ration assistance",
            "Local relief support"
        ],
        "Moderate": [
            "Temporary shelter",
            "Medical camps",
            "Interim cash relief",
            "Livelihood recovery assistance"
        ],
        "Severe": [
            "Immediate evacuation",
            "Disaster compensation",
            "Fast-track pension approval",
            "Housing reconstruction",
            "Long-term rehabilitation"
        ]
    }


# ============================================================
# 5. PENSION POLICY ENGINE
# ============================================================

class PensionPolicyEngine:
    """
    Determines applicable pension & welfare schemes.
    """

    def eligible_schemes(self, age: int | None, facts: Set[str]) -> List[str]:
        schemes = []

        if age is not None and age >= 60:
            schemes.append("Old Age Pension")

        if "widow" in facts:
            schemes.append("Widow Pension")

        if any(f in facts for f in [
            "physical disability",
            "visual disability",
            "hearing disability",
            "severe disability"
        ]):
            schemes.append("Disability Pension")

        if any(f in facts for f in [
            "flood", "cyclone", "fire", "earthquake"
        ]):
            schemes.append("Disaster Relief Assistance")

        if "no income" in facts and age is not None and age < 60:
            schemes.append("Temporary Financial Assistance")

        return list(dict.fromkeys(schemes))

    def pension_types(self, age: int | None, facts: Set[str]) -> List[str]:
        pensions = []

        if age is not None and age >= 60:
            pensions.append("Old Age Pension")

        if "widow" in facts:
            pensions.append("Widow Pension")

        if any(f in facts for f in [
            "physical disability",
            "visual disability",
            "hearing disability",
            "severe disability"
        ]):
            pensions.append("Disability Pension")

        if any(f in facts for f in [
            "flood", "cyclone", "fire", "earthquake"
        ]):
            pensions.append("Disaster Relief Pension")

        return list(dict.fromkeys(pensions))


# ============================================================
# 6. RISK SCORING (FINAL & REALISTIC)
# ============================================================

class RiskScorer:
    """
    Computes final priority using weighted
    vulnerability + disaster severity.
    """

    def score(self, vulnerability: Dict[str, float], severity: str) -> Dict[str, str]:

        vuln_score = (
            vulnerability["economic"] * 0.4 +
            vulnerability["health"] * 0.35 +
            vulnerability["social"] * 0.25
        )

        severity_boost = {
            "None": 0.0,
            "Low": 0.15,
            "Moderate": 0.35,
            "Severe": 0.6
        }[severity]

        final_score = min(vuln_score + severity_boost, 1.0)

        if final_score >= 0.75:
            priority = "High Priority"
        elif final_score >= 0.45:
            priority = "Medium Priority"
        else:
            priority = "Low Priority"

        return {
            "risk_score": round(final_score, 2),
            "priority_level": priority
        }


# ============================================================
# 7. STAGE-2 MASTER ENGINE
# ============================================================

class SemanticPolicyEngine:
    """
    End-to-end Stage-2 policy intelligence engine.
    """

    def __init__(self):
        self.fact_extractor = FactExtractor()
        self.age_analyzer = AgeAnalyzer()
        self.vulnerability_model = VulnerabilityModel()
        self.disaster_ontology = DisasterOntology()
        self.pension_engine = PensionPolicyEngine()
        self.risk_scorer = RiskScorer()

    def analyze(self, normalized_text: str) -> Dict:

        facts = self.fact_extractor.extract(normalized_text)

        age = self.age_analyzer.extract_age(normalized_text)
        age_group = self.age_analyzer.age_group(age)

        vulnerability = self.vulnerability_model.compute(age_group, facts)

        severity = self.disaster_ontology.severity(facts)
        actions = self.disaster_ontology.GOVERNMENT_ACTIONS[severity]

        schemes = self.pension_engine.eligible_schemes(age, facts)
        pensions = self.pension_engine.pension_types(age, facts)

        risk = self.risk_scorer.score(vulnerability, severity)

        return {
            # Backward-compatible fields
            "Age": age,
            "Age_Group": age_group,
            "Extracted_Facts": sorted(facts),
            "Vulnerability_Profile": vulnerability,
            "Disaster_Severity": severity,
            "Government_Response": actions,
            "Eligible_Schemes": schemes,
            "Risk_Assessment": risk,

            # Enhanced pension intelligence
            "Pension_Types": pensions,
            "Primary_Pension": pensions[0] if pensions else None,
            "Disaster_Pension_Flag": (
                "Disaster Relief Pension" in pensions
            )
        }
# ============================================================
# SEER-NLP : STAGE 3 — DECISION, PENSION & PRIORITY ENGINE
# ============================================================
# Converts policy analysis into:
# - Pension decision & processing state
# - Final administrative verdict
# - Priority ranking
# - Explainable justification
# - Audit-ready trace
#
# Deterministic | Transparent | Officer-friendly
# ============================================================

from typing import Dict, List


# ============================================================
# 1. FINAL VERDICT ENGINE
# ============================================================

class DecisionVerdictEngine:
    """
    Determines final administrative verdict.
    """

    def verdict(self, risk_level: str, schemes: List[str]) -> str:

        if risk_level == "High Priority":
            return "IMMEDIATE ACTION REQUIRED"

        if risk_level == "Medium Priority" and schemes:
            return "ELIGIBLE (PROCESS WITH PRIORITY)"

        if risk_level == "Low Priority" and schemes:
            return "ELIGIBLE (LOW PRIORITY)"

        return "INSUFFICIENT EVIDENCE"


# ============================================================
# 2. PENSION STATE RESOLVER
# ============================================================

class PensionStateResolver:
    """
    Determines pension processing state.
    """

    def resolve(self, risk_level: str, pensions: List[str]) -> str:

        if not pensions:
            return "NOT ELIGIBLE"

        if risk_level == "High Priority":
            return "APPROVED (FAST-TRACK)"

        if risk_level == "Medium Priority":
            return "UNDER VERIFICATION"

        return "ELIGIBLE (QUEUE)"


# ============================================================
# 3. PRIORITY QUEUE ENGINE
# ============================================================

class PriorityQueueEngine:
    """
    Numeric ranking for administrative ordering.
    """

    PRIORITY_MAP = {
        "High Priority": 1,
        "Medium Priority": 2,
        "Low Priority": 3
    }

    def priority_rank(self, priority_level: str) -> int:
        return self.PRIORITY_MAP.get(priority_level, 4)


# ============================================================
# 4. EXPLAINABILITY GENERATOR
# ============================================================

class ExplainabilityGenerator:
    """
    Produces concise, officer-readable explanation.
    """

    def generate(
        self,
        age: int | None,
        age_group: str,
        facts: List[str],
        vulnerability: Dict[str, float],
        severity: str,
        pensions: List[str],
        pension_state: str,
        risk: Dict[str, str]
    ) -> str:

        lines = []

        if age:
            lines.append(f"Applicant is {age} years old ({age_group}).")

        if severity != "None":
            lines.append(f"Affected by {severity.lower()} disaster conditions.")

        if pensions:
            lines.append("Identified pension eligibility: " + ", ".join(pensions) + ".")

        if vulnerability["health"] >= 0.5:
            lines.append("Significant health-related vulnerability identified.")

        if vulnerability["economic"] >= 0.5:
            lines.append("Economic distress detected due to income or livelihood loss.")

        if vulnerability["social"] >= 0.5:
            lines.append("Social dependency risk identified.")

        lines.append(f"Pension processing status: {pension_state}.")
        lines.append(f"Overall application priority classified as {risk['priority_level']}.")

        return " ".join(lines)


# ============================================================
# 5. AUDIT TRACE BUILDER
# ============================================================

class AuditTraceBuilder:
    """
    Builds structured, audit-ready reasoning trace.
    """

    def build(
        self,
        part2_output: Dict,
        verdict: str,
        pension_state: str,
        priority_rank: int
    ) -> Dict:

        return {
            "Decision_Verdict": verdict,
            "Pension_State": pension_state,
            "Priority_Rank": priority_rank,
            "Inputs_Used": {
                "Age": part2_output["Age"],
                "Age_Group": part2_output["Age_Group"],
                "Extracted_Facts": part2_output["Extracted_Facts"],
                "Vulnerability_Profile": part2_output["Vulnerability_Profile"],
                "Disaster_Severity": part2_output["Disaster_Severity"],
                "Pension_Types": part2_output.get("Pension_Types", []),
                "Eligible_Schemes": part2_output["Eligible_Schemes"]
            },
            "Risk_Assessment": part2_output["Risk_Assessment"]
        }


# ============================================================
# 6. STAGE-3 MASTER DECISION ENGINE
# ============================================================

class FinalDecisionEngine:
    """
    End-to-end decision, pension & priority engine.
    """

    def __init__(self):
        self.verdict_engine = DecisionVerdictEngine()
        self.pension_state_engine = PensionStateResolver()
        self.priority_engine = PriorityQueueEngine()
        self.explainer = ExplainabilityGenerator()
        self.audit_builder = AuditTraceBuilder()

    def decide(self, part2_output: Dict) -> Dict:

        age = part2_output["Age"]
        age_group = part2_output["Age_Group"]
        facts = part2_output["Extracted_Facts"]
        vulnerability = part2_output["Vulnerability_Profile"]
        severity = part2_output["Disaster_Severity"]
        schemes = part2_output["Eligible_Schemes"]
        pensions = part2_output.get("Pension_Types", [])
        risk = part2_output["Risk_Assessment"]

        verdict = self.verdict_engine.verdict(
            risk["priority_level"], schemes
        )

        pension_state = self.pension_state_engine.resolve(
            risk["priority_level"], pensions
        )

        priority_rank = self.priority_engine.priority_rank(
            risk["priority_level"]
        )

        explanation = self.explainer.generate(
            age, age_group, facts,
            vulnerability, severity,
            pensions, pension_state,
            risk
        )

        audit_trace = self.audit_builder.build(
            part2_output, verdict, pension_state, priority_rank
        )

        return {
            "Final_Verdict": verdict,
            "Priority_Level": risk["priority_level"],
            "Priority_Rank": priority_rank,

            # Pension-centric output
            "Pension": {
                "Types": pensions,
                "Primary": pensions[0] if pensions else None,
                "State": pension_state
            },

            "Explanation": explanation,
            "Government_Response": part2_output["Government_Response"],
            "Eligible_Schemes": schemes,
            "Audit_Trace": audit_trace
        }
# ============================================================
# DATASET-DRIVEN EXECUTION LAYER
# ============================================================
# Purpose:
# - Apply SEER-NLP pipeline on real-world datasets (CSV)
# - Preserve deterministic NLP + rule-based decision flow
# - Enable batch processing for evaluation & reporting
#
# NOTE:
# This section DOES NOT modify any core engine logic.
# It only consumes already-defined classes.
# ============================================================

import pandas as pd


def run_seer_nlp_on_dataset(
    input_csv: str,
    text_column: str,
    output_csv: str = "seer_nlp_results.csv",
    limit: int | None = None
):
    """
    Runs full SEER-NLP pipeline on a CSV dataset.

    Parameters:
    - input_csv   : path to dataset CSV
    - text_column : column containing narrative / application text
    - output_csv  : output CSV path
    - limit       : optional row limit for testing
    """

    # Load dataset
    df = pd.read_csv(input_csv)
    if limit:
        df = df.head(limit)

    # Initialize engines
    normalizer = TextNormalizer()
    policy_engine = SemanticPolicyEngine()
    decision_engine = FinalDecisionEngine()

    results = []

    for idx, row in df.iterrows():
        raw_text = str(row.get(text_column, "")).strip()
        if not raw_text:
            continue

        # ---------------- Phase 1 ----------------
        norm_out = normalizer.normalize(raw_text)

        # ---------------- Phase 2 ----------------
        policy_out = policy_engine.analyze(
            norm_out["normalized_text"]
        )

        # ---------------- Phase 3 ----------------
        final_out = decision_engine.decide(policy_out)

        results.append({
            "Row_ID": idx,
            "Raw_Text": raw_text,
            "Normalized_Text": norm_out["normalized_text"],
            "Noise_Score": norm_out["noise_score"],
            "Age": policy_out["Age"],
            "Age_Group": policy_out["Age_Group"],
            "Extracted_Facts": ", ".join(policy_out["Extracted_Facts"]),
            "Disaster_Severity": policy_out["Disaster_Severity"],
            "Priority_Level": final_out["Priority_Level"],
            "Final_Verdict": final_out["Final_Verdict"],
            "Primary_Pension": final_out["Pension"]["Primary"],
            "Pension_State": final_out["Pension"]["State"]
        })

    # Save output
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"[SEER-NLP] Dataset processing complete → {output_csv}")
