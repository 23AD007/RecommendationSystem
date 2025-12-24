# src/utils/schema_validation.py

class SchemaValidationError(Exception):
    pass


NLP_REQUIRED_COLUMNS = {
    "eco_pressure",
    "cost_efficiency",
    "durability_pressure",
    "innovation_level"
}


def validate_schema(df):
    missing = NLP_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SchemaValidationError(f"Missing NLP columns: {missing}")
