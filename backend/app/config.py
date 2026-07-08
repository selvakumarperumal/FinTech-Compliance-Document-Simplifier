import os
from typing import Optional
from pydantic import SecretStr
from dotenv import load_dotenv

load_dotenv(".env")


class Settings:
    """Application settings using environment variables."""

    def __init__(self):
        google_key = os.getenv("GOOGLE_API_KEY")
        self.google_api_key: Optional[SecretStr] = SecretStr(google_key) if google_key else None
        
        self.google_model_name: str = os.getenv("GOOGLE_MODEL_NAME", "gemini-2.5-flash")
                
        self.environment: str = os.getenv("ENVIRONMENT", "development")
        self.debug: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")


settings = Settings()