import pandas as pd

# =========================================================
# 1. ULTRA-EXPANDED VOCABULARY (DETERMINISTIC)
# =========================================================

AGES = list(range(60, 91))   # 60–90

LOCATIONS = [
    "Gaya district", "Patna district", "Bhagalpur district",
    "Muzaffarpur district", "Darbhanga district",
    "Ranchi city", "Dhanbad city", "Jamshedpur city",
    "Varanasi district", "Prayagraj district", "Kanpur city",
    "Lucknow city", "Bhopal district", "Indore city",
    "Ujjain district", "Jaipur city", "Ajmer district",
    "Kota city", "Cuttack district", "Puri district",
    "Balasore district", "Sambalpur district",
    "Nagpur city", "Aurangabad district", "Amravati district",
    "Kolkata city", "Howrah district", "Midnapore district",
    "Raipur city", "Bilaspur district"
]

DISASTERS = {
    "Flood": [
        "floods damaged my house",
        "water entered my home during floods",
        "my house was submerged due to flooding",
        "flood waters washed away my belongings",
        "severe flooding left my house unlivable",
        "continuous rain caused flooding in my area",
        "flood destroyed my shelter completely"
    ],
    "Cyclone": [
        "cyclone destroyed my shelter",
        "strong winds damaged my house",
        "storm ruined my home",
        "cyclone caused major structural damage",
        "high speed winds blew away my roof",
        "cyclonic storm left my house unsafe",
        "cyclone affected my entire locality"
    ],
    "Fire": [
        "fire destroyed my house",
        "my home was burnt in a fire accident",
        "lost everything due to fire",
        "fire broke out and damaged my shelter",
        "fire accident left my house unsafe",
        "electrical fire burnt my home",
        "fire spread rapidly and destroyed my house"
    ],
    "Earthquake": [
        "earthquake damaged my house",
        "tremors cracked my home",
        "after the earthquake my house became unsafe",
        "earthquake caused my house walls to collapse",
        "continuous tremors weakened my house structure",
        "earthquake shook my home violently",
        "seismic activity damaged my shelter"
    ]
}

PENSION_INTENTS = {
    "OldAge": [
        "because of my age i cannot work",
        "i am too old to earn now",
        "i am elderly and dependent on support",
        "my age does not allow me to do physical work",
        "i have no strength to work at this age",
        "due to old age i am unemployed",
        "my advanced age prevents me from earning"
    ],
    "Widow": [
        "my husband passed away",
        "after losing my husband i live alone",
        "i am a widow with no income support",
        "my husband died and i have no earning source",
        "i lost my spouse and depend on government help",
        "after my spouse passed away i became financially dependent",
        "i am widowed and have no family income"
    ],
    "Disability": [
        "i am disabled and unable to work",
        "medical problems prevent me from working",
        "i suffer from physical disability",
        "my health condition does not allow me to earn",
        "i am medically unfit for work",
        "chronic illness has stopped me from working",
        "i am physically impaired and unemployed"
    ]
}

TEMPLATES = [
    "I am {age} years old and live in {location}. {disaster}. {vulnerability}.",
    "Living in {location}, I am {age} years old. {vulnerability}. {disaster}.",
    "{disaster_cap} in {location}. I am {age} years old. {vulnerability}.",
    "I live in {location}. {disaster}. Due to my condition, {vulnerability}.",
    "At the age of {age}, I stay in {location}. {disaster}. {vulnerability}.",
    "Currently residing in {location}, I am {age} years old. {disaster}. {vulnerability}.",
    "I am {age} years of age. {disaster}. I live in {location} and {vulnerability}.",
    "After the incident, I remain in {location}. {disaster}. At {age}, {vulnerability}.",
    "My age is {age} and I stay in {location}. {disaster}. As a result, {vulnerability}.",
    "I reside in {location}. Following the disaster, {disaster}. At {age}, {vulnerability}."
]

# =========================================================
# 2. DATASET GENERATION (BALANCED & DETERMINISTIC)
# =========================================================

TARGET_SIZE = 10000
records = []

for disaster, disaster_phrases in DISASTERS.items():
    for pension, vulnerability_phrases in PENSION_INTENTS.items():
        for age in AGES:
            for location in LOCATIONS:
                for d_phrase in disaster_phrases:
                    for v_phrase in vulnerability_phrases:
                        for template in TEMPLATES:

                            if len(records) >= TARGET_SIZE:
                                break

                            text = template.format(
                                age=age,
                                location=location,
                                disaster=d_phrase,
                                disaster_cap=d_phrase.capitalize(),
                                vulnerability=v_phrase
                            )

                            records.append({
                                "narrative_text": text,
                                "disaster_type": disaster,
                                "pension_intent": pension
                            })

                        if len(records) >= TARGET_SIZE:
                            break
                    if len(records) >= TARGET_SIZE:
                        break
                if len(records) >= TARGET_SIZE:
                    break
            if len(records) >= TARGET_SIZE:
                break
        if len(records) >= TARGET_SIZE:
            break
    if len(records) >= TARGET_SIZE:
        break

# =========================================================
# 3. SAVE DATASET
# =========================================================

df = pd.DataFrame(records)
df.to_csv("synthetic_pension_disaster_dataset_10k.csv", index=False)

# =========================================================
# 4. VERIFICATION
# =========================================================

print("✅ Dataset generation complete")
print("Total samples:", len(df))
print("\nCounts by Disaster Type:")
print(df["disaster_type"].value_counts())
print("\nCounts by Pension Intent:")
print(df["pension_intent"].value_counts())
print("\n📁 Saved as: synthetic_pension_disaster_dataset_10k.csv")
