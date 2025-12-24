import pandas as pd
from sqlalchemy import create_engine
import os

# Database connection details
DB_URL = "postgresql://postgres:ads%401234@localhost:5432/postgres"

OUTPUT_PATH = "data/processed/training_dataset.csv"


def main():
    engine = create_engine(DB_URL)

    query = """
    SELECT
        product_category,
        fragility_score,
        sustainability_priority,
        durability_requirement,
        max_packaging_cost,
        material_cost,
        innovation_level,
        recommended
    FROM training_dataset
    WHERE recommended IS NOT NULL
    """

    df = pd.read_sql(query, engine)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"training_dataset.csv created")
    print(f"Shape: {df.shape}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
