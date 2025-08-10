import os
from typing import Optional
from pydantic import SecretStr
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv(".env")


class Settings:
    """Application settings using environment variables."""

    def __init__(self):
        # API keys for various services
        gemini_key = os.getenv("GEMINI_API_KEY")
        self.gemini_api_key: Optional[SecretStr] = SecretStr(gemini_key) if gemini_key else None
        
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_api_key: Optional[SecretStr] = SecretStr(anthropic_key) if anthropic_key else None
        
        mistral_key = os.getenv("MISTRAL_API_KEY")
        self.mistral_api_key: Optional[SecretStr] = SecretStr(mistral_key) if mistral_key else None
        
        hf_key = os.getenv("HF_TOKEN")
        self.hf_token: Optional[SecretStr] = SecretStr(hf_key) if hf_key else None
        
        # Database connection URL
        self.database_url: Optional[str] = os.getenv("DATABASE_URL")
         
        # Directory for vector store
        self.vector_dir: str = os.getenv("VECTOR_DIR", "./vector_store")
        
        # Application settings
        self.environment: str = os.getenv("environment", "development")
        self.debug: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")


# Create a global settings instance
settings = Settings()
