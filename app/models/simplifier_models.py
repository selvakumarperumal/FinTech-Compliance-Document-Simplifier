from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.documents import Document


class FileResponse(BaseModel):
    """Simplified response model for content processing results"""
    operation_id: str = Field(description="Unique identifier for this operation")
    content: List[Document] = Field(description="Extracted and processed content from documents")
    model : Optional[str] = Field(None, description="Model used for processing, if applicable")

class LinkResponse(BaseModel):
    """Response model for link processing results"""
    links: List[str] = Field(description="List of processed links")
    content: List[Document] = Field(description="Extracted and processed content from links")
    model: Optional[str] = Field(None, description="Model used for processing, if applicable")

class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str = Field(description="Error message")
    error_type: Optional[str] = Field(None, description="Type of error")
    operation_id: Optional[str] = Field(None, description="Operation ID if available")

class SimplifyResponse(BaseModel):
    """Response model for simplification results"""
    simplified_content: str = Field(description="Simplified content after processing")