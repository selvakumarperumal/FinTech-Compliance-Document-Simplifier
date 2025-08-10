import os
import shutil
from typing import List, Dict, Any
from pathlib import Path
from fastapi import UploadFile
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader
)
import logging

# Configure logging for error tracking
logger = logging.getLogger(__name__)


class FileHandler:
    """Handle file operations for the FinTech Compliance Document Simplifier."""
    
    def __init__(self, base_temp_dir: str = "/tmp/compliance_docs"):
        """
        Initialize FileHandler with a base temporary directory.
        
        Args:
            base_temp_dir: Base directory for temporary file storage
        """
        self.base_temp_dir = base_temp_dir
        self._ensure_temp_dir_exists()
    
    def _ensure_temp_dir_exists(self) -> None:
        """Ensure the base temporary directory exists."""
        try:
            Path(self.base_temp_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create temp directory {self.base_temp_dir}: {str(e)}")
            raise IOError(f"Cannot create temporary directory: {str(e)}")
    
    async def save_uploaded_files(self, files: List[UploadFile], uuid: str) -> Dict[str, Any]:
        """
        Save uploaded files from FastAPI endpoint to temporary folder.
        
        Args:
            files: List of FastAPI UploadFile objects
            uuid: UUID string to use as basename for the temporary directory
            
        Returns:
            Dictionary containing:
            - 'saved_files': List of saved file paths
            - 'temp_dir': Temporary directory path
            - 'file_info': List of file information dictionaries
            
        Raises:
            ValueError: If no files provided or files are invalid
            IOError: If file saving fails
        """
        try:
            if not files:
                raise ValueError("No files provided")
            
            # Create a temporary directory using the provided UUID as basename
            temp_dir = os.path.join(self.base_temp_dir, f"upload_{uuid}")
            os.makedirs(temp_dir, exist_ok=True)
            
            saved_files: List[str] = []
            file_info: List[Dict[str, Any]] = []
            
            try:
                for file in files:
                    if not file.filename:
                        continue
                        
                    # Create safe filename (remove path traversal attempts)
                    safe_filename = self._sanitize_filename(file.filename)
                    file_path = os.path.join(temp_dir, safe_filename)
                    
                    # Save file content
                    with open(file_path, 'wb') as temp_file:
                        # Read file content in chunks to handle large files
                        while chunk := await file.read(8192):  # 8KB chunks
                            temp_file.write(chunk)
                    
                    # Reset file position for potential re-reading
                    await file.seek(0)
                    
                    # Get file information
                    file_stat = os.stat(file_path)
                    file_info.append({
                        "original_name": file.filename,
                        "saved_name": safe_filename,
                        "path": file_path,
                        "size_bytes": file_stat.st_size,
                        "content_type": file.content_type or "unknown"
                    })
                    
                    saved_files.append(file_path)
                    
            except Exception as e:
                # Clean up on error
                self.cleanup_temp_dir(temp_dir)
                logger.error(f"File save error for UUID {uuid}: {str(e)}")
                raise IOError(f"Failed to save files: {str(e)}")
            
            return {
                "saved_files": saved_files,
                "temp_dir": temp_dir,
                "file_info": file_info
            }
        except ValueError as ve:
            logger.error(f"Validation error in save_uploaded_files: {str(ve)}")
            raise
        except IOError as io_error:
            logger.error(f"IO error in save_uploaded_files: {str(io_error)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in save_uploaded_files for UUID {uuid}: {str(e)}")
            raise RuntimeError(f"File upload failed: {str(e)}")
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent path traversal and invalid characters.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        try:
            # Remove path components
            filename = os.path.basename(filename)
            
            # Replace potentially dangerous characters
            dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
            for char in dangerous_chars:
                filename = filename.replace(char, '_')
            
            # Ensure filename is not empty and has reasonable length
            if not filename or len(filename) > 255:
                filename = f"file_{os.urandom(8).hex()}"
                
            return filename
        except Exception as e:
            logger.error(f"Failed to sanitize filename '{filename}': {str(e)}")
            # Return a safe fallback filename
            return f"file_{os.urandom(8).hex()}"
    
    def cleanup_temp_dir(self, temp_dir: str) -> bool:
        """
        Clean up temporary directory and all its contents.
        
        Args:
            temp_dir: Path to temporary directory
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Successfully cleaned up temp directory: {temp_dir}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to cleanup temp directory {temp_dir}: {str(e)}")
            return False

    def cleanup_uploaded_files(self, uuid: str) -> bool:
        """
        Clean up all uploaded files for a specific UUID.
        
        Args:
            uuid: UUID string to identify the upload session
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            temp_dir = os.path.join(self.base_temp_dir, f"upload_{uuid}")
            return self.cleanup_temp_dir(temp_dir)
        except Exception as e:
            logger.error(f"Failed to cleanup uploaded files for UUID {uuid}: {str(e)}")
            return False

    def read_files_from_temp_directory(self, uuid: str) -> Any:
        """
        Read files from a temporary directory and return loaded documents.
        
        Args:
            uuid: UUID string to identify the upload session
        
        Returns:
            List of loaded documents
        """
        try:
            temp_dir = os.path.join(self.base_temp_dir, f"upload_{uuid}")
            if not os.path.exists(temp_dir):
                raise ValueError(f"Temporary directory {temp_dir} does not exist")
            if not os.listdir(temp_dir):
                raise ValueError(f"Temporary directory {temp_dir} is empty")
            
            loaders = {
                '.txt': TextLoader,
                '.pdf': PyPDFLoader,
                '.docx': Docx2txtLoader
            }
            document_names = []
            documents = []
            
            for filename in os.listdir(temp_dir):
                try:
                    document_names.append(filename)
                    file_path = os.path.join(temp_dir, filename)
                    file_extension = os.path.splitext(filename)[1]
                    loader = loaders.get(file_extension, TextLoader)
                    doc = loader(file_path).load()
                    documents.extend(doc)
                except Exception as file_error:
                    logger.error(f"Error loading file {filename} for UUID {uuid}: {str(file_error)}")
                    # Continue processing other files instead of failing completely
                    continue

            return (documents, document_names)
            
        except ValueError as ve:
            logger.error(f"Validation error in read_files_from_temp_directory: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in read_files_from_temp_directory for UUID {uuid}: {str(e)}")
            raise RuntimeError(f"Failed to read files from temp directory: {str(e)}")

# Create a global instance
file_handler = FileHandler()