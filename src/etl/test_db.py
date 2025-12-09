from src.ecopackdb.db_connect import get_engine
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
import pandas as pd
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get values from .env
USER = os.getenv("DB_USER")
PASSWORD = os.getenv("DB_PASSWORD")
HOST = os.getenv("DB_HOST")
PORT = os.getenv("DB_PORT")
DBNAME = "EcoPackAI "

# Encode the password to handle special characters
ENCODED_PASSWORD = quote_plus(PASSWORD)
url = f"postgresql+psycopg2://{USER}:{ENCODED_PASSWORD}@{HOST}:{PORT}/{DBNAME}"

print("Using URL:", url)

try:
    engine = create_engine(url)
    df = pd.read_sql("SELECT current_database() as db;", engine)
    print("Connection Successful! Current database:")
    print(df)
except Exception as e:
    print("Connection Failed:")
    print(e)
