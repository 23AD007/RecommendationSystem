import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DATABASE_URL = os.getenv("DATABASE_URL")
    DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
