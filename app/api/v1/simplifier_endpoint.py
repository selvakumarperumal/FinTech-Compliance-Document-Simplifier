from fastapi import UploadFile, File, Form, APIRouter, HTTPException
from typing import List, Optional
from app.services.simplifier import simplify_content as simplify_service
from app.models.simplifier_models import ErrorResponse, FileResponse, LinkResponse, SimplifyResponse
import uuid
from app.models.anthropic_models import ModelChoices
from app.services.llm_service import simplify_content_service

router = APIRouter()

@router.post("/upload", response_model=SimplifyResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def upload_files_endpoint(
    files: Optional[List[UploadFile]] = File(None),
    uuid_param: Optional[str] = Form(None, alias="uuid"),
    model: Optional[ModelChoices] = Form(None)
):
    """
    Upload and process files for content simplification.
    
    This endpoint:
    1. Receives uploaded files from the client  
    2. Saves them to temporary directory with UUID-based organization
    3. Extracts and processes document content using LangChain loaders
    4. Returns operation ID and extracted content only
    
    Args:
        files: List of files to upload and process (optional)
                Supported formats: PDF, DOCX, TXT
        uuid_param: Optional UUID for the operation (auto-generated if not provided)
        
    Returns:
        SimplifyResponse containing:
        - operation_id: Unique identifier for this processing session
        - content: List of extracted text content from all documents
        
    Raises:
        HTTPException 400: If validation errors occur in the service
        HTTPException 500: If file processing fails
        
    Example Usage:
        curl -X POST "http://localhost:8000/api/v1/upload" \\
             -F "files=@document.pdf" \\
             -F "files=@report.docx" \\
             -F "uuid=my-custom-uuid"
             
    Example Response:
        {
            "operation_id": "uuid-here",
            "content": ["Document text 1...", "Document text 2..."]
        }
    """
    try:
        # Generate UUID if not provided
        operation_uuid = uuid_param or str(uuid.uuid4())
        
        # Call simplifier service with files (let service handle validation)
        result = await simplify_service(files=files, links=None, uuid=operation_uuid)
        
        print(result)
        
        file_response = FileResponse(
            operation_id=operation_uuid,
            content=result.get("documents", []),
            model=model.value if model else None
        )

        print(f"FileResponse created with operation_id: {file_response.operation_id}, model: {file_response.model}")

        response = await simplify_content_service(content=file_response.content, model=file_response.model)

        # Return simplified response with just operation_id and content
        return response

    except HTTPException:
        # Re-raise HTTPExceptions as-is (400, 500, etc.)
        raise
    except ValueError as e:
        # Handle validation errors from the service
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        # Handle any other unexpected errors
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


@router.post("/links", response_model=SimplifyResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def process_links_endpoint(
    links: Optional[str] = Form(None),
    uuid_param: Optional[str] = Form(None, alias="uuid"),
    model : Optional[ModelChoices] = Form(None)
):
    """
    Process links/URLs for content simplification.
    
    This endpoint:
    1. Receives URLs or text containing URLs from the client
    2. Downloads and saves content to temporary directory with UUID-based organization
    3. Extracts and processes web content using LangChain loaders
    4. Returns operation ID and extracted content only
    
    Args:
        links: URLs or text containing URLs to process (optional)
               Can be single URL, multiple URLs separated by spaces/newlines, or text with embedded URLs
        uuid_param: Optional UUID for the operation (auto-generated if not provided)
        
    Returns:
        SimplifyResponse containing:
        - operation_id: Unique identifier for this processing session
        - content: List of extracted text content from all web pages
        
    Raises:
        HTTPException 400: If validation errors occur in the service
        HTTPException 500: If link processing fails
        
    Example Usage:
        curl -X POST "http://localhost:8000/api/v1/links" \\
             -F "links=https://example.com/doc1 https://example.com/doc2" \\
             -F "uuid=my-custom-uuid"
             
    Example Response:
        {
            "operation_id": "uuid-here",
            "content": ["Web page text 1...", "Web page text 2..."]
        }
    """
    try:
        # Generate UUID if not provided
        operation_uuid = uuid_param or str(uuid.uuid4())
        
        # Call simplifier service with links (let service handle validation)
        result = await simplify_service(files=None, links=links, uuid=operation_uuid)
        
        
        # Return simplified response with just operation_id and content
        file_response = LinkResponse(
            links=result.get("links", []),
            content=result.get("documents", []),
            model=model.value if model else None
        )
    
        response = await simplify_content_service(content=file_response.content, model=file_response.model)

        return response

    except HTTPException:
        # Re-raise HTTPExceptions as-is (400, 500, etc.)
        raise
    except ValueError as e:
        # Handle validation errors from the service
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        # Handle any other unexpected errors
        raise HTTPException(status_code=500, detail=f"Link processing failed: {str(e)}")
  
