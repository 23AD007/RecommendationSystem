from src.ecopackdb.db_connect import get_engine
import pandas as pd
def load_data():
    engine = get_engine()
    # Adjust table name as needed; assuming 'engineered_options' exists in the DB
    return pd.read_sql("SELECT * FROM engineered_options", engine)