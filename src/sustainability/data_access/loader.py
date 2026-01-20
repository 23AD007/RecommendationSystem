from src.ecopackdb.db_connect import get_engine
import pandas as pd
def load_data():
    return pd.read_csv("data/processed/engineered_options.csv")