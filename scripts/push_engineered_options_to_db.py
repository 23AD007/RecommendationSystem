import pandas as pd
from sqlalchemy import create_engine
from src.ecopackdb.db_connect import get_engine

# Path to your CSV file
CSV_PATH = "data/processed/engineered_options.csv"
# Name of the table to create in PostgreSQL
table_name = "engineered_options"

def main():
    # Load CSV
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    # Get SQLAlchemy engine
    engine = get_engine()

    # Push to PostgreSQL (replace if exists)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Table '{table_name}' created/replaced in the database.")

if __name__ == "__main__":
    main()
