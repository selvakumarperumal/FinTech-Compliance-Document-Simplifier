import os
from typing import List, Dict, Any
from urllib.parse import urlparse
import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import Document
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LinkHandler:
    """Handle web link operations for the FinTech Compliance Document Simplifier."""
    
    def __init__(self, base_temp_dir: str = "/tmp/compliance_docs"):
        """
        Initialize LinkHandler with a base temporary directory.
        
        Args:
            base_temp_dir: Base directory for temporary file storage
        """
        self.base_temp_dir = base_temp_dir
        self.supported_schemes = ['http', 'https']
    
    def validate_url(self, url: str) -> bool:
        """
        Validate if the URL is properly formatted and accessible.
        
        Args:
            url: URL string to validate
            
        Returns:
            True if URL is valid, False otherwise
        """
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                logger.warning(f"Invalid URL format: {url}")
                return False
            
            if parsed.scheme not in self.supported_schemes:
                logger.warning(f"Unsupported URL scheme '{parsed.scheme}' for URL: {url}")
                return False
                
            # Test if URL is accessible
            response = requests.head(url, timeout=10, allow_redirects=True)
            return response.status_code < 400
            
        except requests.RequestException as e:
            logger.error(f"Network error validating URL {url}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error validating URL {url}: {str(e)}")
            return False
    
    def load_web_content(self, urls: List[str]) -> Dict[str, Any]:
        """
        Load content from web URLs using LangChain WebBaseLoader.
        
        Args:
            urls: List of URLs to load content from
            
        Returns:
            Dictionary containing:
            - 'documents': List of loaded document objects
            - 'successful_urls': List of successfully loaded URLs
            - 'failed_urls': List of URLs that failed to load
            - 'metadata': Metadata about the loading process
            
        Raises:
            ValueError: If no valid URLs provided
        """
        try:
            logger.info(f"Loading content from {len(urls)} URLs")

            
            if not urls:
                raise ValueError("No URLs provided")
            
            # Validate URLs
            valid_urls = []
            invalid_urls = []
            
            for url in urls:
                if self.validate_url(url):
                    valid_urls.append(url)
                else:
                    invalid_urls.append(url)
            
            if not valid_urls:
                raise ValueError("No valid URLs provided")
            
            documents: List[Document] = []
            successful_urls: List[str] = []
            
            for url in valid_urls:
                try:
                    # Create WebBaseLoader for the URL
                    loader = WebBaseLoader(url)
                    
                    # Load documents from the URL
                    url_documents = loader.load()
                    
                    
                    documents.extend(url_documents)
                    successful_urls.append(url)
                    
                except Exception as e:
                    logger.error(f"Failed to load content from URL {url}: {str(e)}")
            
            return {
                'documents': documents,
                'successful_urls': successful_urls,
            }
        

        except ValueError as ve:
            logger.error(f"Validation error in load_web_content: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in load_web_content: {str(e)}")
            raise RuntimeError(f"Failed to load web content: {str(e)}")
    
    def load_single_url(self, url: str) -> Dict[str, Any]:
        """
        Load content from a single URL.
        
        Args:
            url: Single URL to load content from
            
        Returns:
            Dictionary containing document and metadata
        """
        try:
            result = self.load_web_content([url])
            
            return {
                'document': result['documents'][0] if result['documents'] else None,
                'success': len(result['successful_urls']) > 0,
            }
        except Exception as e:
            logger.error(f"Failed to load single URL {url}: {str(e)}")
            raise RuntimeError(f"Failed to load single URL: {str(e)}")
    
    def extract_links_from_content(self, content: str) -> List[str]:
        """
        Extract HTTP/HTTPS links from text content.
        
        Args:
            content: Text content to extract links from
            
        Returns:
            List of extracted URLs
        """
        try:
            # Regex pattern to find URLs
            url_pattern = r'https?://[^\s<>"]{2,}'
            urls = re.findall(url_pattern, content)
            
            # Validate extracted URLs
            valid_urls = []
            for url in urls:
                if self.validate_url(url):
                    valid_urls.append(url)
            
            logger.info(f"Extracted {len(valid_urls)} valid URLs from {len(urls)} found URLs")
            return valid_urls
        except re.error as e:
            logger.error(f"Regex error in extract_links_from_content: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in extract_links_from_content: {str(e)}")
            return []
    
    def save_web_content_to_temp(self, urls: List[str], uuid: str) -> Dict[str, Any]:
        """
        Load web content and save it to temporary directory.
        
        Args:
            urls: List of URLs to load content from
            uuid: UUID string to use as basename for the temporary directory
            
        Returns:
            Dictionary containing loaded documents and file paths
        """
        try:
            # Extract links from content if provided as text
            if isinstance(urls, str):
                urls = self.extract_links_from_content(urls)
            
            # Load web content
            result = self.load_web_content(urls)
            
            # Create temporary directory
            temp_dir = os.path.join(self.base_temp_dir, f"web_content_{uuid}")
            os.makedirs(temp_dir, exist_ok=True)
            
            saved_files: List[str] = []
            
            try:
                for i, doc in enumerate(result['documents']):
                    # Create filename from URL
                    source_url = result.get('successful_urls', [])[i] 
                    safe_filename = self._url_to_filename(source_url, i)
                    file_path = os.path.join(temp_dir, f"{safe_filename}.txt")
                    
                    # Save content to file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(f"Source URL: {source_url}\n")
                        f.write(f"Content Length: {len(doc.page_content)}\n")
                        f.write(f"{'='*50}\n\n")
                        f.write(doc.page_content)
                    
                    saved_files.append(file_path)
                    
            except Exception as e:
                # Clean up on error
                import shutil
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                logger.error(f"Failed to save web content files: {str(e)}")
                raise IOError(f"Failed to save web content: {str(e)}")
            
            return {
                'documents': result['documents'],
                'successful_urls': result['successful_urls']
            }
        except ValueError as ve:
            logger.error(f"Validation error in save_web_content_to_temp: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in save_web_content_to_temp for UUID {uuid}: {str(e)}")
            raise RuntimeError(f"Failed to save web content to temp: {str(e)}")
    
    def _url_to_filename(self, url: str, index: int) -> str:
        """
        Convert URL to a safe filename.
        
        Args:
            url: URL to convert
            index: Index for uniqueness
            
        Returns:
            Safe filename string
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace('.', '_')
            path = parsed.path.replace('/', '_').replace('.', '_')
            filename = f"{domain}{path}_{index}"
        except Exception as e:
            logger.error(f"Failed to parse URL for filename generation {url}: {str(e)}")
            filename = f"web_content_{index}"
        
        try:
            # Remove dangerous characters
            dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
            for char in dangerous_chars:
                filename = filename.replace(char, '_')
            
            # Limit length
            if len(filename) > 200:
                filename = filename[:200]
                
            return filename
        except Exception as e:
            logger.error(f"Failed to sanitize filename for URL {url}: {str(e)}")
            return f"web_content_{index}"
    
    def read_files_from_temp_directory(self, uuid: str) -> Any:
        """
        Read files from a temporary directory and return loaded documents.
        
        Args:
            uuid: UUID string to identify the upload session
            
        Returns:
            List of loaded documents
        """
        try:
            temp_dir = os.path.join(self.base_temp_dir, f"web_content_{uuid}")
            
            if not os.path.exists(temp_dir):
                raise FileNotFoundError(f"Temporary directory {temp_dir} does not exist")
            
            documents = []
            document_names = []
            for filename in os.listdir(temp_dir):
                document_names.append(filename)
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            documents.append(Document(page_content=content, metadata={'source_file': filename}))
                    except Exception as e:
                        logger.error(f"Failed to read file {file_path}: {str(e)}")
                        continue

            return documents, document_names
        except FileNotFoundError as fe:
            logger.error(f"Directory not found in read_files_from_temp_directory for UUID {uuid}: {str(fe)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in read_files_from_temp_directory for UUID {uuid}: {str(e)}")
            raise RuntimeError(f"Failed to read files from temp directory: {str(e)}")

# Create a global instance
link_handler = LinkHandler()
