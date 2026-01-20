import os

class Config:
    DEBUG = False
    API_KEY = os.getenv("API_KEY", "packaging-api-key-2024")
