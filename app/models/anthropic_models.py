"""
Anthropic Models Demo - Simple Model Listing

This script uses the anthropic package to list all available models.
"""

import anthropic
import os
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
from app.config import settings
from enum import Enum


# Ensure the Anthropic API key is set
ANTHROPIC_API_KEY = settings.anthropic_api_key

if not ANTHROPIC_API_KEY:
    raise ValueError("Anthropic API key is not set or invalid")


class ModelInfo(BaseModel):
    """Pydantic model for Anthropic model information"""
    id: str
    created_at: Optional[datetime]
    display_name: str
    type: str


class ModelsResponse(BaseModel):
    """Pydantic model for the models list response"""
    data: List[ModelInfo]
    has_more: bool
    first_id: Optional[str] = None
    last_id: Optional[str] = None


class AnthropicModelsDemo:
    """Demo class for listing Anthropic models"""

    def __init__(self, api_key: str):
        """Initialize with Anthropic client"""
        self.client = anthropic.Anthropic(api_key=api_key)
        if not self.client:
            raise ValueError("Anthropic API key is not set or invalid")
        print("Anthropic client initialized successfully")

    def list_models(self, limit: int = 20) -> ModelsResponse:
        """List all available models with structured response"""
        response = self.client.models.list(limit=limit)
        
        # Convert the response to our structured format
        models_data: List[ModelInfo] = []
        for model in response.data:
            model_info = ModelInfo(
                id=model.id,
                created_at=model.created_at,
                display_name=model.display_name,
                type=model.type
            )
            models_data.append(model_info)
        
        return ModelsResponse(
            data=models_data,
            has_more=response.has_more or False,
            first_id=response.first_id,
            last_id=response.last_id
        )
    
    def clean_enum_key(self, name: str) -> str:
        """Clean model name for enum key"""
        return name.lower().replace(" ", "_").replace(".", "_").replace("(", "").replace(")", "")


    def get_available_model_names(self, limit: int = 20) -> Dict[str, str]:
        """Get available models from Anthropic"""
        models_response = self.list_models(limit=limit)
        display_names = {self.clean_enum_key(model.display_name): model.id for model in models_response.data}
        return display_names

ModelChoices = Enum('ModelChoices', AnthropicModelsDemo(ANTHROPIC_API_KEY.get_secret_value()).get_available_model_names(), type=str)
