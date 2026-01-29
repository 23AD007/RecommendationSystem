from ecopackdb.db_connect import get_engine
import pandas as pd

try:
    engine = get_engine()
    df = pd.read_sql(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public';",
        engine,
    )
    print("Connected! Tables in DB:")
    print(df)
except Exception as e:
    print("Connection Failed:")
    print(e)
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in the .env file.")

print("Using URL:", DATABASE_URL)

try:
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("SELECT current_database() as db;", engine)
    print("Connection Successful! Current database:")
    print(df)
except Exception as e:
    print("Connection Failed:")
    print(e)
