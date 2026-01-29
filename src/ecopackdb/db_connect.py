# src/ecopackdb/db_connect.py

from urllib.parse import quote_plus
from sqlalchemy import create_engine

# === UPDATE ONLY THESE VALUES === #
DB_USER = "postgres"
DB_PASSWORD = "ads@1234"      
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "postgres"          



def get_engine():
    """
    Build and return a SQLAlchemy Engine connected to the PostgreSQL database.
    """
    user = DB_USER
    password = quote_plus(DB_PASSWORD)
    host = DB_HOST
    port = DB_PORT
    dbname = DB_NAME

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    print("DEBUG URL:", url)

    engine = create_engine(url)
    return engine
