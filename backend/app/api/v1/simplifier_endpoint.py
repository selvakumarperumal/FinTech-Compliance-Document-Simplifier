from fastapi import UploadFile, File, Form, APIRouter, HTTPException
from fastapi.responses import Response
from typing import List, Optional
from app.services.simplifier import simplify_content as simplify_service
import uuid
import logging
from app.services.llm_service import simplify_content_service
from app.services.pdf_service import generate_pdf_from_content
from app.services.file_handler import file_handler

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload")
async def upload_files_endpoint(
    files: List[UploadFile] = File(...),
    uuid_param: Optional[str] = Form(None, alias="uuid"),
    response_format: Optional[str] = Form("pdf", alias="response_format"),
    model: Optional[str] = Form(None)
):
    """
    Upload files, simplify content, and download as PDF or return JSON text.
    
    Args:
        files: List of files to upload (PDF, DOCX, TXT)
        uuid_param: Optional UUID for the operation
        response_format: Target output format ('pdf' or 'json')
        model: Optional NVIDIA LLM model name override
        
    Returns:
        Downloadable PDF file or JSON object containing simplified content
    """
    operation_uuid = uuid_param or str(uuid.uuid4())
    try:
        result = await simplify_service(files=files, uuid=operation_uuid)
        
        documents = result.get("documents", [])
        response = await simplify_content_service(content=documents, model=model)
        simplified_text = response["simplified_content"]
        
        if response_format == "json":
            return {
                "uuid": operation_uuid,
                "simplified_content": simplified_text
            }
            
        pdf_bytes = generate_pdf_from_content(simplified_text)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=simplified_{operation_uuid}.pdf"}
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        try:
            file_handler.cleanup_uploaded_files(operation_uuid)
        except Exception as cleanup_error:
            logger.error(f"Failed to clean up files for UUID {operation_uuid}: {str(cleanup_error)}")


@router.post("/generate-pdf")
async def generate_pdf_endpoint(
    content: str = Form(...),
    title: Optional[str] = Form("Simplified Compliance Document")
):
    """
    Generate PDF from simplified text content.
    
    Args:
        content: The simplified text content to write to PDF
        title: Optional title for the PDF header
        
    Returns:
        Downloadable PDF document bytes
    """
    try:
        pdf_bytes = generate_pdf_from_content(content, title)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=simplified_document.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for status monitoring."""
    return {"status": "ok"}
