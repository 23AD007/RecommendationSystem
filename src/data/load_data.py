import pandas as pd
from sqlalchemy import create_engine
from src.utils.config import DATABASE_URL

def load_training_data():
    engine = create_engine(DATABASE_URL)

    query = """
    SELECT
        fragility_score,
        sustainability_priority,
        durability_requirement,
        max_packaging_cost,
        material_cost,
        innovation_level,
        overall_sustainability_score
    FROM training_dataset
    WHERE overall_sustainability_score IS NOT NULL
    """

    return pd.read_sql(query, engine)
