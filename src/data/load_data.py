from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from src.config import Config

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(Config.DATABASE_URL, pool_pre_ping=True)
    return _engine


def test_db_connection():
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError as e:
        print("Database connection error:", e)
        return False
