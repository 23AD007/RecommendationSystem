import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DATABASE_URL = os.getenv("DATABASE_URL")
    DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"

    # 🔐 API Key for securing endpoints
    API_KEY = os.getenv("API_KEY", "packaging-api-key-2024")
