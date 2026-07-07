import os
from typing import Optional
from pydantic import SecretStr
from dotenv import load_dotenv

load_dotenv(".env")


class Settings:
    """Application settings using environment variables."""

    def __init__(self):
        nvidia_key = os.getenv("NVIDIA_API_KEY")
        self.nvidia_api_key: Optional[SecretStr] = SecretStr(nvidia_key) if nvidia_key else None
        
        self.nvidia_model_name: str = os.getenv("NVIDIA_MODEL_NAME", "minimaxai/minimax-m3")
                
        self.environment: str = os.getenv("ENVIRONMENT", "development")
        self.debug: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")


settings = Settings()