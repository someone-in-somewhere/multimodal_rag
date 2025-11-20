"""
Document Parser
Parses various document formats (PDF, DOCX, HTML, Images, Text)
"""

import io
import base64
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import mimetypes

import pypdf
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
from PIL import Image
from pdf2image import convert_from_bytes
from tabulate import tabulate

from config import settings

logger = logging.getLogger(__name__)

# Try to import pytesseract (OCR)
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    logger.info("✅ Tesseract OCR available")
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("⚠️  Tesseract OCR not available (pip install pytesseract)")


class DocumentParser:
    """
    Parse various document formats and extract structured content
    
    Supported formats:
    - PDF (.pdf)
    - Word Documents (.docx, .doc)
    - HTML (.html, .htm)
    - Plain Text (.txt, .md)
    - Images (.png, .jpg, .jpeg, .gif, .webp)
    """
    
    def __init__(self):
        """Initialize the parser"""
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.figures_dir = settings.FIGURES_DIR
        
        # Create figures directory
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"DocumentParser initialized "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
    
    async def parse_document(
        self,
        content: bytes,
        filename: str,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parse a document and extract structured content
        
        Args:
            content: File content as bytes
            filename: Original filename
            content_type: MIME type (optional)
        
        Returns:
            Dictionary with:
            - text_chunks: List of text chunks
            - tables: List of table dictionaries
            - images: List of image dictionaries
        """
        # Determine content type
        if not content_type:
            content_type, _ = mimetypes.guess_type(filename)
        
        logger.info(f"Parsing {filename} (type: {content_type})")
        
        # Route to appropriate parser
        if content_type == "application/pdf" or filename.endswith('.pdf'):
            return await self._parse_pdf(content, filename)
        
        elif content_type in [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword"
        ] or filename.endswith(('.docx', '.doc')):
            return await self._parse_docx(content, filename)
        
        elif content_type == "text/html" or filename.endswith(('.html', '.htm')):
            return await self._parse_html(content, filename)
        
        elif content_type and content_type.startswith('image/'):
            return await self._parse_image(content, filename)
        
        elif content_type and content_type.startswith('text/'):
            return await self._parse_text(content, filename)
        
        else:
            # Try to parse as text
            logger.warning(f"Unknown content type: {content_type}, trying as text")
            return await self._parse_text(content, filename)
    
    async def _parse_pdf(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Parse PDF file"""
        logger.info(f"📄 Parsing PDF: {filename}")
        
        try:
            # Extract text using pypdf
            pdf_reader = pypdf.PdfReader(io.BytesIO(content))
            
            full_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n\n"
            
            # Convert PDF pages to images
            images = []
            try:
                pdf_images = convert_from_bytes(content, dpi=200)
                
                for idx, img in enumerate(pdf_images):
                    img_filename = f"{Path(filename).stem}_page_{idx}.png"
                    img_path = self.figures_dir / img_filename
                    
                    # Save image
                    img.save(img_path, "PNG")
                    
                    # Convert to base64
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    images.append({
                        'id': f"page_{idx}",
                        'path': str(img_path),
                        'base64': img_base64
                    })
                
                logger.info(f"✅ Extracted {len(images)} pages as images")
            
            except Exception as e:
                logger.warning(f"Failed to convert PDF to images: {e}")
            
            # Chunk text
            text_chunks = self._chunk_text(full_text) if full_text else []
            
            logger.info(
                f"✅ PDF parsed: {len(text_chunks)} text chunks, {len(images)} images"
            )
            
            return {
                'text_chunks': text_chunks,
                'tables': [],  # Table extraction from PDF is complex, skipping for now
                'images': images
            }
        
        except Exception as e:
            logger.error(f"❌ PDF parsing failed: {e}")
            raise
    
    async def _parse_docx(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Parse DOCX file"""
        logger.info(f"📝 Parsing DOCX: {filename}")
        
        try:
            doc = DocxDocument(io.BytesIO(content))
            
            # Extract text from paragraphs
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            full_text = "\n\n".join(paragraphs)
            
            # Extract tables
            tables = []
            for idx, table in enumerate(doc.tables):
                table_data = []
                
                for row in table.rows:
                    row_data = [cell.text.strip() for cell in row.cells]
                    table_data.append(row_data)
                
                if table_data:
                    # Convert to markdown format using tabulate
                    try:
                        if len(table_data) > 1:
                            markdown_table = tabulate(
                                table_data[1:],  # Data rows
                                headers=table_data[0],  # Header row
                                tablefmt="github"
                            )
                        else:
                            markdown_table = tabulate(table_data, tablefmt="github")
                        
                        tables.append({
                            'id': f"table_{idx}",
                            'content': markdown_table
                        })
                    except Exception as e:
                        logger.warning(f"Failed to format table {idx}: {e}")
            
            # Extract images (more complex, skipping for now)
            images = []
            
            # Chunk text
            text_chunks = self._chunk_text(full_text) if full_text else []
            
            logger.info(
                f"✅ DOCX parsed: {len(text_chunks)} text chunks, {len(tables)} tables"
            )
            
            return {
                'text_chunks': text_chunks,
                'tables': tables,
                'images': images
            }
        
        except Exception as e:
            logger.error(f"❌ DOCX parsing failed: {e}")
            raise
    
    async def _parse_html(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Parse HTML file"""
        logger.info(f"🌐 Parsing HTML: {filename}")
        
        try:
            # Decode content
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('latin-1')
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            full_text = soup.get_text(separator='\n\n')
            
            # Extract tables
            tables = []
            for idx, table in enumerate(soup.find_all('table')):
                table_data = []
                
                for row in table.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    row_data = [cell.get_text(strip=True) for cell in cells]
                    if row_data:
                        table_data.append(row_data)
                
                if table_data:
                    try:
                        markdown_table = tabulate(
                            table_data[1:] if len(table_data) > 1 else table_data,
                            headers=table_data[0] if len(table_data) > 1 else [],
                            tablefmt="github"
                        )
                        tables.append({
                            'id': f"table_{idx}",
                            'content': markdown_table
                        })
                    except Exception as e:
                        logger.warning(f"Failed to format HTML table {idx}: {e}")
            
            # Chunk text
            text_chunks = self._chunk_text(full_text) if full_text else []
            
            logger.info(
                f"✅ HTML parsed: {len(text_chunks)} text chunks, {len(tables)} tables"
            )
            
            return {
                'text_chunks': text_chunks,
                'tables': tables,
                'images': []
            }
        
        except Exception as e:
            logger.error(f"❌ HTML parsing failed: {e}")
            raise
    
    async def _parse_text(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Parse plain text file"""
        logger.info(f"📃 Parsing text: {filename}")
        
        try:
            # Try UTF-8 first, fallback to latin-1
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('latin-1', errors='ignore')
            
            # Chunk text
            text_chunks = self._chunk_text(text) if text else []
            
            logger.info(f"✅ Text parsed: {len(text_chunks)} chunks")
            
            return {
                'text_chunks': text_chunks,
                'tables': [],
                'images': []
            }
        
        except Exception as e:
            logger.error(f"❌ Text parsing failed: {e}")
            raise
    
    async def _parse_image(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Parse image file"""
        logger.info(f"🖼️  Parsing image: {filename}")
        
        try:
            # Open image
            img = Image.open(io.BytesIO(content))
            
            # Save to figures directory
            img_filename = Path(filename).name
            img_path = self.figures_dir / img_filename
            img.save(img_path)
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format=img.format or "PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Try OCR if available
            text_chunks = []
            if TESSERACT_AVAILABLE:
                try:
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        text_chunks = self._chunk_text(ocr_text)
                        logger.info(f"✅ OCR extracted {len(text_chunks)} text chunks")
                except Exception as e:
                    logger.warning(f"OCR failed: {e}")
            
            logger.info(f"✅ Image parsed: {img_path}")
            
            return {
                'text_chunks': text_chunks,
                'tables': [],
                'images': [{
                    'id': 'image_0',
                    'path': str(img_path),
                    'base64': img_base64
                }]
            }
        
        except Exception as e:
            logger.error(f"❌ Image parsing failed: {e}")
            raise
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into smaller pieces with overlap
        
        Args:
            text: Text to chunk
        
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            
            # Get chunk
            chunk = text[start:end]
            
            # If not at the end, try to find a good breaking point
            if end < text_length:
                # Look for sentence boundaries
                last_period = chunk.rfind('. ')
                last_newline = chunk.rfind('\n')
                last_question = chunk.rfind('? ')
                last_exclamation = chunk.rfind('! ')
                
                # Find the best boundary
                boundary = max(last_period, last_newline, last_question, last_exclamation)
                
                # Only break at boundary if it's in the second half of the chunk
                if boundary > self.chunk_size // 2:
                    chunk = chunk[:boundary + 1]
                    end = start + boundary + 1
            
            chunk = chunk.strip()
            
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
        
        logger.debug(f"Chunked text into {len(chunks)} chunks")
        
        return chunks
    
    def cleanup_old_figures(self, days: int = 7):
        """
        Clean up old figure files
        
        Args:
            days: Delete files older than this many days
        """
        import time
        from datetime import datetime, timedelta
        
        cutoff = time.time() - (days * 86400)
        deleted = 0
        
        for file_path in self.figures_dir.iterdir():
            if file_path.is_file():
                if file_path.stat().st_mtime < cutoff:
                    try:
                        file_path.unlink()
                        deleted += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
        
        if deleted > 0:
            logger.info(f"🧹 Cleaned up {deleted} old figure files")
