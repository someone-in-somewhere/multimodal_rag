import os
import base64
import uuid
from pathlib import Path
from typing import List, Dict, Any
from io import BytesIO

import pypdf
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
from tabulate import tabulate
import structlog

from config import settings

logger = structlog.get_logger()

class DocumentParser:
    """
    Parses various document formats (PDF, DOCX, HTML, TXT) and extracts:
    - Text chunks
    - Tables (as Markdown)
    - Images (saved to figures/ directory)
    """
    
    def __init__(self):
        self.figures_dir = settings.FIGURES_DIR
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
    
    async def parse_document(
        self,
        file_content: bytes,
        filename: str,
        file_type: str
    ) -> Dict[str, Any]:
        """
        Main parsing method that routes to appropriate parser.
        """
        logger.info(f"Parsing document: {filename} (type: {file_type})")
        
        if file_type == "application/pdf":
            return await self._parse_pdf(file_content, filename)
        elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            return await self._parse_docx(file_content, filename)
        elif file_type == "text/html":
            return await self._parse_html(file_content, filename)
        elif file_type.startswith("image/"):
            return await self._parse_image(file_content, filename)
        elif file_type in ["text/plain", "text/markdown", "application/octet-stream"]:
            return await self._parse_text(file_content, filename)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    async def _parse_pdf(self, content: bytes, filename: str) -> Dict[str, Any]:
        text_chunks = []
        tables = []
        images = []
        
        try:
            pdf_reader = pypdf.PdfReader(BytesIO(content))
            full_text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                full_text += page_text + "\n\n"
            
            text_chunks = self._chunk_text(full_text)
            
            pdf_images = convert_from_bytes(content)
            for idx, img in enumerate(pdf_images):
                img_id = f"{Path(filename).stem}_page_{idx}_{uuid.uuid4().hex[:8]}"
                img_path = self.figures_dir / f"{img_id}.png"
                img.save(img_path)
                
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                images.append({
                    'id': img_id,
                    'path': str(img_path),
                    'base64': img_base64
                })
            
            logger.info(f"Parsed PDF: {len(text_chunks)} chunks, {len(images)} images")
            
        except Exception as e:
            logger.error(f"Error parsing PDF: {str(e)}")
            raise
        
        return {
            'text_chunks': text_chunks,
            'tables': tables,
            'images': images,
            'metadata': {'filename': filename, 'type': 'pdf'}
        }
    
    async def _parse_docx(self, content: bytes, filename: str) -> Dict[str, Any]:
        text_chunks = []
        tables = []
        images = []
        
        try:
            doc = DocxDocument(BytesIO(content))
            
            full_text = "\n".join([para.text for para in doc.paragraphs])
            text_chunks = self._chunk_text(full_text)
            
            for table_idx, table in enumerate(doc.tables):
                table_data = []
                for row in table.rows:
                    table_data.append([cell.text for cell in row.cells])
                
                if table_data:
                    markdown_table = tabulate(table_data[1:], headers=table_data[0], tablefmt="github")
                    tables.append({
                        'content': markdown_table,
                        'raw': str(table_data),
                        'id': f"table_{table_idx}"
                    })
            
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    img_id = f"{Path(filename).stem}_{uuid.uuid4().hex[:8]}"
                    img_path = self.figures_dir / f"{img_id}.png"
                    
                    img_data = rel.target_part.blob
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                    
                    img_base64 = base64.b64encode(img_data).decode()
                    
                    images.append({
                        'id': img_id,
                        'path': str(img_path),
                        'base64': img_base64
                    })
            
            logger.info(f"Parsed DOCX: {len(text_chunks)} chunks, {len(tables)} tables, {len(images)} images")
            
        except Exception as e:
            logger.error(f"Error parsing DOCX: {str(e)}")
            raise
        
        return {
            'text_chunks': text_chunks,
            'tables': tables,
            'images': images,
            'metadata': {'filename': filename, 'type': 'docx'}
        }
    
    async def _parse_html(self, content: bytes, filename: str) -> Dict[str, Any]:
        text_chunks = []
        tables = []
        
        try:
            soup = BeautifulSoup(content, 'lxml')
            text = soup.get_text(separator='\n')
            text_chunks = self._chunk_text(text)
            
            for table_idx, table in enumerate(soup.find_all('table')):
                table_data = []
                for row in table.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    table_data.append([cell.get_text(strip=True) for cell in cells])
                
                if table_data:
                    markdown_table = tabulate(table_data[1:], headers=table_data[0], tablefmt="github") if len(table_data) > 1 else str(table_data)
                    tables.append({
                        'content': markdown_table,
                        'raw': str(table_data),
                        'id': f"table_{table_idx}"
                    })
            
            logger.info(f"Parsed HTML: {len(text_chunks)} chunks, {len(tables)} tables")
            
        except Exception as e:
            logger.error(f"Error parsing HTML: {str(e)}")
            raise
        
        return {
            'text_chunks': text_chunks,
            'tables': tables,
            'images': [],
            'metadata': {'filename': filename, 'type': 'html'}
        }
    
    async def _parse_image(self, content: bytes, filename: str) -> Dict[str, Any]:
        try:
            img = Image.open(BytesIO(content))
            img_id = f"{Path(filename).stem}_{uuid.uuid4().hex[:8]}"
            img_path = self.figures_dir / f"{img_id}.png"
            img.save(img_path)
            
            img_base64 = base64.b64encode(content).decode()
            ocr_text = pytesseract.image_to_string(img)
            text_chunks = self._chunk_text(ocr_text) if ocr_text.strip() else []
            
            logger.info(f"Parsed image: {filename}")
            
            return {
                'text_chunks': text_chunks,
                'tables': [],
                'images': [{
                    'id': img_id,
                    'path': str(img_path),
                    'base64': img_base64
                }],
                'metadata': {'filename': filename, 'type': 'image'}
            }
        except Exception as e:
            logger.error(f"Error parsing image: {str(e)}")
            raise
    
    async def _parse_text(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Parse plain text or markdown file."""
        try:
            # Try UTF-8 first, fallback to latin-1
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('latin-1')
            
            # Chunk text
            text_chunks = self._chunk_text(text)
            
            logger.info(f"Parsed text file: {len(text_chunks)} chunks")
            
            return {
                'text_chunks': text_chunks,
                'tables': [],
                'images': [],
                'metadata': {'filename': filename, 'type': 'text'}
            }
        except Exception as e:
            logger.error(f"Error parsing text: {str(e)}")
            raise
    
    def _chunk_text(self, text: str) -> List[str]:
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if end < text_length:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                boundary = max(last_period, last_newline)
                if boundary > self.chunk_size // 2:
                    chunk = chunk[:boundary + 1]
                    end = start + boundary + 1
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        
        return [c for c in chunks if c]