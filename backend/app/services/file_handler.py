import os
import shutil
from typing import List, Dict, Any, Tuple
from pathlib import Path
from fastapi import UploadFile
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
import logging

logger = logging.getLogger(__name__)


class FileHandler:
    """Handle file operations for the FinTech Compliance Document Simplifier."""
    
    def __init__(self, base_temp_dir: str = "/tmp/compliance_docs"):
        self.base_temp_dir = base_temp_dir
        self._ensure_temp_dir_exists()
    
    def _ensure_temp_dir_exists(self) -> None:
        """Ensure the base temporary directory exists."""
        Path(self.base_temp_dir).mkdir(parents=True, exist_ok=True)
    
    async def save_uploaded_files(self, files: List[UploadFile], uuid: str) -> Dict[str, Any]:
        """Save uploaded files from FastAPI endpoint to temporary folder."""
        if not files:
            raise ValueError("No files provided")
        
        temp_dir = os.path.join(self.base_temp_dir, f"upload_{uuid}")
        os.makedirs(temp_dir, exist_ok=True)
        
        saved_files: List[str] = []
        file_info: List[Dict[str, Any]] = []
        
        try:
            for file in files:
                if not file.filename:
                    continue
                    
                safe_filename = self._sanitize_filename(file.filename)
                file_path = os.path.join(temp_dir, safe_filename)
                
                with open(file_path, 'wb') as temp_file:
                    while chunk := await file.read(8192):
                        temp_file.write(chunk)
                
                await file.seek(0)
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
            self.cleanup_temp_dir(temp_dir)
            raise IOError(f"Failed to save files: {str(e)}")
        
        return {"saved_files": saved_files, "temp_dir": temp_dir, "file_info": file_info}
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal and invalid characters."""
        filename = os.path.basename(filename)
        dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        if not filename or len(filename) > 255:
            filename = f"file_{os.urandom(8).hex()}"
        return filename
    
    def cleanup_temp_dir(self, temp_dir: str) -> bool:
        """Clean up temporary directory and all its contents."""
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to cleanup temp directory {temp_dir}: {str(e)}")
            return False

    def cleanup_uploaded_files(self, uuid: str) -> bool:
        """Clean up all uploaded files for a specific UUID."""
        temp_dir = os.path.join(self.base_temp_dir, f"upload_{uuid}")
        return self.cleanup_temp_dir(temp_dir)

    def read_files_from_temp_directory(self, uuid: str) -> Tuple[List[Any], List[str]]:
        """Read files from a temporary directory and return loaded documents."""
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
        
        documents = []
        document_names = []
        
        for filename in os.listdir(temp_dir):
            try:
                document_names.append(filename)
                file_path = os.path.join(temp_dir, filename)
                file_extension = os.path.splitext(filename)[1]
                loader = loaders.get(file_extension, TextLoader)
                doc = loader(file_path).load()
                documents.extend(doc)
            except Exception as e:
                logger.error(f"Error loading file {filename}: {str(e)}")
                continue

        return documents, document_names


file_handler = FileHandler()