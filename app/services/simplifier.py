from app.services.file_handler import file_handler
from app.services.link_handler import link_handler
from fastapi import UploadFile
from typing import Optional, List, Dict, Any, Union
import uuid as uuid_module
import logging

# Configure logging for error tracking
logger = logging.getLogger(__name__)

# This module handles the links and files loading processes
# and simplifies the content based on the provided files or links.
# and this is not api route, so no need to import FastAPI or APIRouter here.
async def simplify_content(files: Optional[List[UploadFile]] = None, links: Optional[Union[List[str], str]] = None, uuid: Optional[str] = None) -> Dict[str, Any]:
    """
    Simplify content from uploaded files or provided links.
    
    Args:
        files: List of uploaded files (optional)
        links: List of URLs to simplify (optional)
        uuid: Unique identifier for the operation (optional)
        
    Returns:
        Dictionary with simplified content and metadata.
        
    Raises:
        ValueError: If no files or links provided
        IOError: If file saving fails
    """
    try:
        if not files and not links:
            raise ValueError("No files or links provided")
        
        # Generate UUID if not provided
        if not uuid:
            uuid = str(uuid_module.uuid4())
        
        result_data: Dict[str, Any] = {
            "documents": [],
        }

        if links:
            result_data["links"] = []
        
        # Handle file uploads
        if files:
            try:
                file_result = await file_handler.save_uploaded_files(files, uuid)
                result_data["file_info"] = file_result["file_info"]
                
                # Read documents from saved files
                (file_documents, _) = file_handler.read_files_from_temp_directory(uuid)
                result_data["documents"].extend(file_documents)
            except Exception as file_error:
                logger.error(f"File processing error for UUID {uuid}: {str(file_error)}")
                raise IOError(f"Failed to process files: {str(file_error)}")
        
        # Handle link processing
        if links:
            try:
                link_result = link_handler.save_web_content_to_temp(links, uuid)
                result_data["links"].extend(link_result["successful_urls"])

                # Read documents from saved links
                link_documents, _ = link_handler.read_files_from_temp_directory(uuid)
                result_data["documents"].extend(link_documents)
            except Exception as link_error:
                logger.error(f"Link processing error for UUID {uuid}: {str(link_error)}")
                raise IOError(f"Failed to process links: {str(link_error)}")

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
    