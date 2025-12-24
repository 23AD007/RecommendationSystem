"""
Single source of truth for ALL feature names used across:
- NLP
- Database
- Training
- Inference
"""

# Canonical feature names (what models expect)
FEATURE_COLUMNS = [
    "eco_pressure",
    "cost_efficiency",
    "durability_pressure",
    "innovation_level",
    "material_cost",
    "fragility_score",
    "max_packaging_cost",
    "durability_requirement",
    "sustainability_priority",
]

# Optional / metadata columns (never used for training)
META_COLUMNS = [
    "recommended",
    "data_source",
    "ingested_at",
]

# Mapping from alternate / legacy column names → canonical names
FEATURE_NAME_MAP = {
    # environmental features
    "environmental_pressure": "eco_pressure",
    "eco_impact": "eco_pressure",

    # cost features
    "cost_efficiency_index": "cost_efficiency",

    # durability
    "durability_score": "durability_pressure",

    # innovation
    "innovation": "innovation_level",
}
