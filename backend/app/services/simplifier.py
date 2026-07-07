from app.services.file_handler import file_handler
from fastapi import UploadFile
from typing import Optional, List, Dict, Any
import uuid as uuid_module
import logging

# Configure logging for error tracking
logger = logging.getLogger(__name__)


async def simplify_content(files: List[UploadFile], uuid: Optional[str] = None) -> Dict[str, Any]:
    """
    Simplify content from uploaded files.
    
    Args:
        files: List of uploaded files
        uuid: Unique identifier for the operation (optional)
        
    Returns:
        Dictionary with simplified content and metadata.
        
    Raises:
        ValueError: If no files provided
        IOError: If file saving fails
    """
    try:
        if not files:
            raise ValueError("No files provided")
        
        # Generate UUID if not provided
        if not uuid:
            uuid = str(uuid_module.uuid4())
        
        result_data: Dict[str, Any] = {
            "documents": [],
        }
        
        # Handle file uploads
        try:
            file_result = await file_handler.save_uploaded_files(files, uuid)
            result_data["file_info"] = file_result["file_info"]
            
            # Read documents from saved files
            (file_documents, _) = file_handler.read_files_from_temp_directory(uuid)
            result_data["documents"].extend(file_documents)
        except Exception as file_error:
            logger.error(f"File processing error for UUID {uuid}: {str(file_error)}")
            raise IOError(f"Failed to process files: {str(file_error)}")
        
        return result_data
        
    except ValueError as ve:
        logger.error(f"Validation error in simplify_content: {str(ve)}")
        raise
    except IOError as io_error:
        logger.error(f"IO error in simplify_content: {str(io_error)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in simplify_content for UUID {uuid}: {str(e)}")
        raise RuntimeError(f"Content simplification failed: {str(e)}")
    