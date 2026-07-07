"""PDF generation service for simplified compliance documents."""

from fpdf import FPDF
from io import BytesIO
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PDFService:
    """Generate PDF documents from simplified content."""
    
    def __init__(self):
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
    
    def generate_pdf(self, content: str, title: str = "Simplified Compliance Document") -> bytes:
        """
        Generate a PDF from simplified content.
        
        Args:
            content: The simplified text content
            title: Document title
            
        Returns:
            PDF file as bytes
        """
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Add title
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, title, ln=True, align="C")
        pdf.ln(5)
        
        # Add timestamp
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
        pdf.ln(10)
        
        # Add content
        pdf.set_font("Helvetica", "", 11)
        
        # Split content into paragraphs and add them
        paragraphs = content.split('\n')
        for paragraph in paragraphs:
            text = paragraph.strip()
            if text:
                # Basic markdown cleaning for a professional PDF appearance
                text = text.replace("**", "")
                text = text.replace("__", "")
                text = text.replace("`", "")
                if text.startswith("#"):
                    text = text.lstrip("#").strip()
                
                # Handle long text with multi_cell for word wrapping
                pdf.multi_cell(0, 6, text)
                pdf.ln(3)
        
        # Return as bytes
        return bytes(pdf.output())


def generate_pdf_from_content(content: str, title: str = "Simplified Compliance Document") -> bytes:
    """Helper function to generate PDF from content."""
    service = PDFService()
    return service.generate_pdf(content, title)
