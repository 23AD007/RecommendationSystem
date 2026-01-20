from sqlalchemy import create_engine
from dotenv import load_dotenv
from urllib.parse import quote_plus
import os

load_dotenv()

def get_engine():
    user = os.getenv("DB_USER")
    password = quote_plus(os.getenv("DB_PASSWORD"))
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    db = os.getenv("DB_NAME")

    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)
