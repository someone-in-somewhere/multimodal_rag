"""
parser_v2.py - Production-Ready RAG Document Parser
==================================================
Enhanced version with:
- ✅ Rich metadata tracking
- ✅ Semantic-aware chunking
- ✅ Table & image extraction
- ✅ Document structure preservation
- ✅ Async operations
- ✅ Full error handling
- ✅ Memory optimization

Supports: PDF, DOCX, HTML, TXT/Markdown, Images (OCR)
"""

import io
import base64
import logging
import re
import hashlib
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import mimetypes
import asyncio

import pypdf
import pdfplumber
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
from PIL import Image
from pdf2image import convert_from_bytes
from tabulate import tabulate

from config import settings

logger = logging.getLogger(__name__)

# ==================== Optional Dependencies ====================

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    logger.info("✅ Tesseract OCR available")
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("⚠️ Tesseract OCR not available")

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
    logger.info("✅ aiohttp available (async HTTP)")
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("⚠️ aiohttp not available, HTML image fetching will be slower")

try:
    # Try new package first (lightweight)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
    logger.info("✅ LangChain text splitter available")
except ImportError:
    try:
        # Fallback to full langchain
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        LANGCHAIN_AVAILABLE = True
        logger.info("✅ LangChain (full) available")
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        logger.warning("⚠️ LangChain not available, using basic chunking")


# ==================== Main Parser Class ====================

class DocumentParser:
    """
    Enhanced document parser with RAG-optimized features
    
    Features:
    - Multi-format support (PDF, DOCX, HTML, TXT, Images)
    - Rich metadata extraction
    - Semantic chunking (with LangChain)
    - Table extraction with structure preservation
    - Image extraction with optional OCR
    - Async operations for performance
    - Document structure tracking (pages, sections, headings)
    """
    
    def __init__(
        self,
        figures_dir: Path = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        enable_semantic_chunking: bool = True,
        enable_table_summarization: bool = False,
        enable_image_description: bool = False,
        max_pdf_pages_as_images: int = 50,
        max_table_summaries: int = 20,
        llm_adapter = None,
        mllm_adapter = None,
    ):
        """
        Initialize DocumentParser
        
        Args:
            figures_dir: Directory to save extracted images
            chunk_size: Size of text chunks (characters)
            chunk_overlap: Overlap between chunks
            enable_semantic_chunking: Use LangChain for smart chunking
            enable_table_summarization: Generate table summaries with LLM
            enable_image_description: Generate image descriptions with MLLM
            max_pdf_pages_as_images: Max PDF pages to convert to images
            max_table_summaries: Max tables to summarize (avoid too many LLM calls)
            llm_adapter: LLM adapter for table summarization (optional)
            mllm_adapter: Multimodal LLM for image description (optional)
        """
        # Config from settings or defaults
        self.figures_dir = figures_dir or settings.FIGURES_DIR
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        # Validate config
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        
        # Feature flags
        self.enable_semantic_chunking = enable_semantic_chunking and LANGCHAIN_AVAILABLE
        self.enable_table_summarization = enable_table_summarization
        self.enable_image_description = enable_image_description
        self.max_pdf_pages_as_images = max_pdf_pages_as_images
        self.max_table_summaries = max_table_summaries
        
        # Optional adapters (will be lazy-loaded if needed)
        self.llm_adapter = llm_adapter
        self.mllm_adapter = mllm_adapter
        
        # Create figures directory
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize semantic splitter if available
        if self.enable_semantic_chunking:
            self.semantic_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )
            logger.info("✅ Semantic chunking enabled (LangChain)")
        else:
            self.semantic_splitter = None
            logger.info("ℹ️ Using basic chunking (LangChain not available)")
        
        logger.info(
            f"DocumentParser v2 initialized:\n"
            f"  - Chunk: size={self.chunk_size}, overlap={self.chunk_overlap}\n"
            f"  - Semantic chunking: {self.enable_semantic_chunking}\n"
            f"  - Table summarization: {enable_table_summarization}\n"
            f"  - Image description: {enable_image_description}\n"
            f"  - Max PDF pages as images: {max_pdf_pages_as_images}"
        )

    # ====================== Main Entry Point ======================
    
    async def parse_document(
        self,
        content: bytes,
        filename: str,
        content_type: Optional[str] = None,
        base_url: Optional[str] = None,
        skip_pdf_images: bool = False,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parse document and extract structured content with rich metadata
        
        Args:
            content: File content as bytes
            filename: Original filename
            content_type: MIME type (auto-detect if None)
            base_url: Base URL for resolving relative links (HTML only)
            skip_pdf_images: Skip converting PDF pages to images (save memory)
            doc_id: Custom document ID (auto-generate if None)
        
        Returns:
            {
                'doc_id': 'unique_id',
                'filename': 'example.pdf',
                'doc_type': 'pdf',
                'metadata': {...},
                'text_chunks': [
                    {
                        'chunk_id': 'doc_123_chunk_0_a1b2',
                        'content': '...',
                        'metadata': {...}
                    }
                ],
                'tables': [...],
                'images': [...],
                'document_structure': {...}
            }
        """
        # Determine content type
        if not content_type:
            content_type, _ = mimetypes.guess_type(filename)
        
        # Generate unique document ID
        if not doc_id:
            doc_id = self._generate_doc_id(content, filename)
        
        logger.info(f"📄 Parsing {filename} (type: {content_type}, id: {doc_id})")
        
        # Route to appropriate parser
        try:
            if content_type == "application/pdf" or filename.endswith('.pdf'):
                result = await self._parse_pdf(content, filename, skip_images=skip_pdf_images)
                doc_type = 'pdf'
            elif content_type in [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword"
            ] or filename.endswith(('.docx', '.doc')):
                result = await self._parse_docx(content, filename)
                doc_type = 'docx'
            elif content_type == "text/html" or filename.endswith(('.html', '.htm')):
                result = await self._parse_html(content, filename, base_url)
                doc_type = 'html'
            elif content_type and content_type.startswith('image/'):
                result = await self._parse_image(content, filename)
                doc_type = 'image'
            elif content_type and content_type.startswith('text/'):
                result = await self._parse_text(content, filename)
                doc_type = 'text'
            else:
                logger.warning(f"Unknown content type: {content_type}, trying as text")
                result = await self._parse_text(content, filename)
                doc_type = 'text'
        except Exception as e:
            logger.error(f"❌ Failed to parse {filename}: {e}")
            raise
        
        # Add document-level metadata
        result['doc_id'] = doc_id
        result['filename'] = filename
        result['doc_type'] = doc_type
        result['metadata'] = {
            'parsed_at': datetime.utcnow().isoformat(),
            'content_type': content_type,
            'size_bytes': len(content),
            'chunk_count': len(result.get('text_chunks', [])),
            'table_count': len(result.get('tables', [])),
            'image_count': len(result.get('images', [])),
        }
        
        # Enrich chunks with document metadata
        result['text_chunks'] = self._enrich_chunks_metadata(
            result.get('text_chunks', []),
            doc_id=doc_id,
            filename=filename,
            doc_type=doc_type
        )
        
        logger.info(
            f"✅ Parsed {filename}: "
            f"{len(result['text_chunks'])} chunks, "
            f"{len(result.get('tables', []))} tables, "
            f"{len(result.get('images', []))} images"
        )
        
        return result

    # ====================== PDF Parser ======================
    
    async def _parse_pdf(
        self,
        content: bytes,
        filename: str,
        skip_images: bool = False
    ) -> Dict[str, Any]:
        """Parse PDF with page-level metadata tracking"""
        logger.info(f"📄 Parsing PDF: {filename}")
        
        text_chunks = []
        tables = []
        images = []
        document_structure = {'pages': [], 'outline': []}
        
        try:
            pdf_reader = pypdf.PdfReader(io.BytesIO(content))
            total_pages = len(pdf_reader.pages)
            
            logger.info(f"PDF has {total_pages} pages")
            
            # Extract outline/bookmarks if available
            try:
                outline = pdf_reader.outline
                document_structure['outline'] = self._extract_pdf_outline(outline)
                logger.debug(f"Extracted {len(document_structure['outline'])} outline items")
            except Exception as e:
                logger.debug(f"No outline available: {e}")
            
            # Extract text page by page (preserve page info)
            page_texts = []
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        page_texts.append({
                            'page': page_num + 1,
                            'text': page_text.strip()
                        })
                        document_structure['pages'].append({
                            'page': page_num + 1,
                            'char_count': len(page_text)
                        })
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
            
            logger.info(f"Extracted text from {len(page_texts)} pages")
            
            # Chunk text with page metadata
            if page_texts:
                text_chunks = await self._chunk_text_with_pages(
                    page_texts,
                    filename=filename,
                    total_pages=total_pages
                )
            
            # Extract tables with pdfplumber
            tables = await self._extract_pdf_tables(content, filename)
            
            # Extract images (optional, memory-intensive)
            if not skip_images:
                images = await self._extract_pdf_images(content, filename, total_pages)
            else:
                logger.info("Skipping PDF image extraction (skip_pdf_images=True)")
        
        except Exception as e:
            logger.error(f"❌ PDF parsing failed for {filename}: {e}")
            raise
        
        return {
            'text_chunks': text_chunks,
            'tables': tables,
            'images': images,
            'document_structure': document_structure
        }
    
    async def _extract_pdf_tables(self, content: bytes, filename: str) -> List[Dict[str, Any]]:
        """Extract tables with rich metadata"""
        tables = []
        
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                table_count = 0
                summarized_count = 0
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_tables = page.extract_tables()
                        
                        for table_idx, table_data in enumerate(page_tables):
                            if not table_data:
                                continue
                            
                            # Clean table data
                            cleaned_table = []
                            for row in table_data:
                                if not row or not any(row):
                                    continue
                                cleaned_row = [
                                    str(cell).strip() if cell else "" 
                                    for cell in row
                                ]
                                if any(cleaned_row):  # Skip empty rows
                                    cleaned_table.append(cleaned_row)
                            
                            if not cleaned_table or len(cleaned_table) < 2:
                                continue
                            
                            # Extract headers and data
                            headers = cleaned_table[0]
                            data_rows = cleaned_table[1:]
                            
                            # Format as markdown
                            try:
                                markdown_table = tabulate(
                                    data_rows,
                                    headers=headers,
                                    tablefmt="github"
                                )
                            except Exception as e:
                                logger.warning(f"Failed to format table on page {page_num + 1}: {e}")
                                markdown_table = str(cleaned_table)
                            
                            # Build table object with rich metadata
                            table_obj = {
                                'id': f"table_{table_count}",
                                'content': markdown_table,
                                'raw_data': cleaned_table,
                                'metadata': {
                                    'source': filename,
                                    'page': page_num + 1,
                                    'table_index': table_idx,
                                    'row_count': len(data_rows),
                                    'col_count': len(headers),
                                    'has_headers': True,
                                    'columns': headers,
                                }
                            }
                            
                            # Generate table summary (with rate limit)
                            if (self.enable_table_summarization and 
                                summarized_count < self.max_table_summaries):
                                table_obj['summary'] = await self._generate_table_summary(
                                    cleaned_table, headers
                                )
                                summarized_count += 1
                            else:
                                table_obj['summary'] = self._generate_simple_table_summary(
                                    cleaned_table, headers
                                )
                            
                            tables.append(table_obj)
                            table_count += 1
                    
                    except Exception as e:
                        logger.warning(f"Failed to extract tables from page {page_num + 1}: {e}")
            
            logger.info(f"✅ Extracted {len(tables)} tables from PDF")
        
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
        
        return tables
    
    async def _extract_pdf_images(
        self,
        content: bytes,
        filename: str,
        total_pages: int
    ) -> List[Dict[str, Any]]:
        """Extract PDF pages as images with memory optimization"""
        images = []
        
        # Limit number of pages to convert (avoid OOM)
        pages_to_convert = min(total_pages, self.max_pdf_pages_as_images)
        
        if total_pages > self.max_pdf_pages_as_images:
            logger.warning(
                f"PDF has {total_pages} pages, only converting first "
                f"{self.max_pdf_pages_as_images} to images"
            )
        
        try:
            # Convert in batches to reduce memory usage
            batch_size = 10
            
            for start_page in range(0, pages_to_convert, batch_size):
                end_page = min(start_page + batch_size, pages_to_convert)
                
                logger.debug(f"Converting PDF pages {start_page + 1}-{end_page} to images")
                
                try:
                    pdf_images = convert_from_bytes(
                        content,
                        dpi=200,
                        first_page=start_page + 1,  # 1-indexed
                        last_page=end_page
                    )
                    
                    for idx, img in enumerate(pdf_images):
                        page_num = start_page + idx
                        
                        img_filename = f"{Path(filename).stem}_page_{page_num}.png"
                        img_path = self.figures_dir / img_filename
                        
                        # Save image
                        img.save(img_path, "PNG")
                        
                        # Convert to base64
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        
                        # Optional: OCR for text extraction
                        ocr_text = ""
                        if TESSERACT_AVAILABLE:
                            try:
                                ocr_text = pytesseract.image_to_string(img)
                            except Exception as e:
                                logger.debug(f"OCR failed for page {page_num + 1}: {e}")
                        
                        image_obj = {
                            'id': f"page_{page_num}",
                            'path': str(img_path),
                            'base64': img_base64,
                            'metadata': {
                                'source': filename,
                                'page': page_num + 1,
                                'format': 'PNG',
                                'size': img.size,
                                'mode': img.mode,
                                'ocr_text': ocr_text.strip() if ocr_text else None,
                            }
                        }
                        
                        # Optional: Generate image description
                        if self.enable_image_description:
                            image_obj['description'] = await self._generate_image_description(img)
                        
                        images.append(image_obj)
                    
                    # Clear memory
                    del pdf_images
                
                except Exception as e:
                    logger.warning(f"Failed to convert PDF pages {start_page + 1}-{end_page}: {e}")
            
            logger.info(f"✅ Extracted {len(images)} PDF pages as images")
        
        except Exception as e:
            logger.warning(f"PDF image conversion failed: {e}")
        
        return images

    # ====================== DOCX Parser ======================
    
    async def _parse_docx(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Parse DOCX with structure preservation"""
        logger.info(f"📝 Parsing DOCX: {filename}")
        
        text_chunks = []
        tables = []
        images = []
        document_structure = {'headings': [], 'sections': []}
        
        try:
            doc = DocxDocument(io.BytesIO(content))
            
            # Extract text with heading structure
            structured_content = []
            current_section = {'heading': None, 'level': 0, 'content': []}
            
            for para in doc.paragraphs:
                if not para.text.strip():
                    continue
                
                # Detect headings
                if para.style.name.startswith('Heading'):
                    # Save previous section
                    if current_section['content']:
                        structured_content.append(current_section.copy())
                    
                    # Start new section
                    try:
                        level = int(para.style.name.replace('Heading', '').strip() or '0')
                    except ValueError:
                        level = 0
                    
                    current_section = {
                        'heading': para.text.strip(),
                        'level': level,
                        'content': []
                    }
                    document_structure['headings'].append({
                        'text': para.text.strip(),
                        'level': level
                    })
                else:
                    current_section['content'].append(para.text)
            
            # Add last section
            if current_section['content']:
                structured_content.append(current_section)
            
            logger.info(f"Extracted {len(structured_content)} sections from DOCX")
            
            # Chunk text with section context
            if structured_content:
                text_chunks = await self._chunk_structured_content(
                    structured_content,
                    filename=filename
                )
            
            # Extract tables
            table_count = 0
            summarized_count = 0
            
            for idx, table in enumerate(doc.tables):
                try:
                    table_data = [
                        [cell.text.strip() for cell in row.cells]
                        for row in table.rows
                    ]
                    
                    if not table_data or len(table_data) < 2:
                        continue
                    
                    headers = table_data[0]
                    data_rows = table_data[1:]
                    
                    try:
                        markdown_table = tabulate(
                            data_rows,
                            headers=headers,
                            tablefmt="github"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to format DOCX table {idx}: {e}")
                        markdown_table = str(table_data)
                    
                    table_obj = {
                        'id': f"table_{table_count}",
                        'content': markdown_table,
                        'raw_data': table_data,
                        'metadata': {
                            'source': filename,
                            'table_index': idx,
                            'row_count': len(data_rows),
                            'col_count': len(headers),
                            'columns': headers,
                        }
                    }
                    
                    # Generate summary (with rate limit)
                    if (self.enable_table_summarization and 
                        summarized_count < self.max_table_summaries):
                        table_obj['summary'] = await self._generate_table_summary(
                            table_data, headers
                        )
                        summarized_count += 1
                    else:
                        table_obj['summary'] = self._generate_simple_table_summary(
                            table_data, headers
                        )
                    
                    tables.append(table_obj)
                    table_count += 1
                
                except Exception as e:
                    logger.warning(f"Failed to extract DOCX table {idx}: {e}")
            
            logger.info(f"Extracted {len(tables)} tables from DOCX")
            
            # Extract images
            image_count = 0
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        image_data = rel.target_part.blob
                        ext = rel.target_part.content_type.split('/')[-1].replace("jpeg", "jpg")
                        
                        img_filename = f"{Path(filename).stem}_image_{image_count}.{ext}"
                        img_path = self.figures_dir / img_filename
                        
                        with open(img_path, 'wb') as f:
                            f.write(image_data)
                        
                        img_base64 = base64.b64encode(image_data).decode()
                        
                        image_obj = {
                            'id': f"image_{image_count}",
                            'path': str(img_path),
                            'base64': img_base64,
                            'metadata': {
                                'source': filename,
                                'index': image_count,
                                'format': ext.upper(),
                            }
                        }
                        
                        # Optional: Generate description
                        if self.enable_image_description:
                            try:
                                img_pil = Image.open(io.BytesIO(image_data))
                                image_obj['description'] = await self._generate_image_description(img_pil)
                            except Exception as e:
                                logger.warning(f"Failed to generate image description: {e}")
                        
                        images.append(image_obj)
                        image_count += 1
                    
                    except Exception as e:
                        logger.warning(f"DOCX image extraction failed: {e}")
            
            logger.info(f"Extracted {len(images)} images from DOCX")
        
        except Exception as e:
            logger.error(f"❌ DOCX parsing failed: {e}")
            raise
        
        return {
            'text_chunks': text_chunks,
            'tables': tables,
            'images': images,
            'document_structure': document_structure
        }

    # ====================== HTML Parser ======================
    
    async def _parse_html(
        self,
        content: bytes,
        filename: str,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Parse HTML with async image fetching"""
        logger.info(f"🌐 Parsing HTML: {filename}")
        
        text_chunks = []
        tables = []
        images = []
        document_structure = {'headings': []}
        
        try:
            # Decode content
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('latin-1', errors='ignore')
            
            soup = BeautifulSoup(text, 'html.parser')
            
            # Remove script and style tags
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()
            
            # Extract headings for structure
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                level = int(heading.name[1])
                document_structure['headings'].append({
                    'text': heading.get_text(strip=True),
                    'level': level
                })
            
            # Extract text
            full_text = soup.get_text(separator="\n\n")
            text_chunks = await self._chunk_text_simple(
                full_text,
                source_metadata={'filename': filename, 'doc_type': 'html'}
            )
            
            # Extract tables
            table_count = 0
            summarized_count = 0
            
            for idx, table in enumerate(soup.find_all('table')):
                try:
                    table_data = []
                    for row in table.find_all('tr'):
                        cells = row.find_all(['td', 'th'])
                        row_data = [cell.get_text(strip=True) for cell in cells]
                        if row_data and any(row_data):
                            table_data.append(row_data)
                    
                    if not table_data or len(table_data) < 2:
                        continue
                    
                    headers = table_data[0]
                    data_rows = table_data[1:]
                    
                    try:
                        markdown_table = tabulate(data_rows, headers=headers, tablefmt="github")
                    except Exception as e:
                        logger.warning(f"HTML table formatting failed: {e}")
                        markdown_table = str(table_data)
                    
                    table_obj = {
                        'id': f"table_{table_count}",
                        'content': markdown_table,
                        'raw_data': table_data,
                        'metadata': {
                            'source': filename,
                            'table_index': idx,
                            'row_count': len(data_rows),
                            'col_count': len(headers),
                            'columns': headers,
                        }
                    }
                    
                    # Generate summary (with rate limit)
                    if (self.enable_table_summarization and 
                        summarized_count < self.max_table_summaries):
                        table_obj['summary'] = await self._generate_table_summary(
                            table_data, headers
                        )
                        summarized_count += 1
                    else:
                        table_obj['summary'] = self._generate_simple_table_summary(
                            table_data, headers
                        )
                    
                    tables.append(table_obj)
                    table_count += 1
                
                except Exception as e:
                    logger.warning(f"Failed to extract HTML table {idx}: {e}")
            
            # Extract images (async if aiohttp available)
            images = await self._extract_html_images(soup, filename, base_url)
        
        except Exception as e:
            logger.error(f"❌ HTML parsing failed: {e}")
            raise
        
        return {
            'text_chunks': text_chunks,
            'tables': tables,
            'images': images,
            'document_structure': document_structure
        }
    
    async def _extract_html_images(
        self,
        soup: BeautifulSoup,
        filename: str,
        base_url: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Extract images from HTML with async fetching"""
        images = []
        image_count = 0
        
        if AIOHTTP_AVAILABLE:
            # Async version (faster)
            images = await self._extract_html_images_async(soup, filename, base_url)
        else:
            # Sync version (fallback)
            images = await self._extract_html_images_sync(soup, filename, base_url)
        
        logger.info(f"✅ Extracted {len(images)} images from HTML")
        return images
    
    async def _extract_html_images_async(
        self,
        soup: BeautifulSoup,
        filename: str,
        base_url: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Async image extraction (requires aiohttp)"""
        import aiohttp
        from urllib.parse import urljoin
        
        images = []
        image_count = 0
        
        async def fetch_image(session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
            """Fetch image data from URL"""
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=10),
                    allow_redirects=True
                ) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        logger.debug(f"Failed to fetch {url}: HTTP {response.status}")
            except asyncio.TimeoutError:
                logger.debug(f"Timeout fetching {url}")
            except aiohttp.ClientError as e:
                logger.debug(f"Client error fetching {url}: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error fetching {url}: {e}")
            
            return None
        
        # Collect all image tasks
        image_tasks = []
        img_tags = []
        
        async with aiohttp.ClientSession() as session:
            for img_tag in soup.find_all('img'):
                img_src = img_tag.get('src')
                if not img_src:
                    continue
                
                img_tags.append(img_tag)
                
                # Handle different image sources
                if img_src.startswith("data:image"):
                    # Base64 embedded image
                    try:
                        header, img_base64_data = img_src.split(",", 1)
                        img_data = base64.b64decode(img_base64_data)
                        image_tasks.append(asyncio.create_task(asyncio.sleep(0, result=img_data)))
                    except Exception as e:
                        logger.warning(f"Failed to decode base64 image: {e}")
                        image_tasks.append(asyncio.create_task(asyncio.sleep(0, result=None)))
                
                elif img_src.startswith(("http://", "https://")):
                    # Absolute URL
                    image_tasks.append(asyncio.create_task(fetch_image(session, img_src)))
                
                elif base_url:
                    # Relative URL
                    img_url = urljoin(base_url, img_src)
                    image_tasks.append(asyncio.create_task(fetch_image(session, img_url)))
                
                else:
                    image_tasks.append(asyncio.create_task(asyncio.sleep(0, result=None)))
            
            # Wait for all images
            if image_tasks:
                img_data_list = await asyncio.gather(*image_tasks, return_exceptions=True)
                
                for img_tag, img_data in zip(img_tags, img_data_list):
                    if isinstance(img_data, Exception) or not img_data:
                        continue
                    
                    try:
                        # Determine extension
                        ext = "png"
                        img_src = img_tag.get('src', '')
                        if 'jpg' in img_src or 'jpeg' in img_src:
                            ext = 'jpg'
                        elif 'gif' in img_src:
                            ext = 'gif'
                        
                        # Save image
                        img_filename = f"{Path(filename).stem}_image_{image_count}.{ext}"
                        img_path = self.figures_dir / img_filename
                        
                        with open(img_path, 'wb') as f:
                            f.write(img_data)
                        
                        img_base64 = base64.b64encode(img_data).decode()
                        
                        # Get alt text
                        alt_text = img_tag.get('alt', '').strip()
                        
                        image_obj = {
                            'id': f"image_{image_count}",
                            'path': str(img_path),
                            'base64': img_base64,
                            'metadata': {
                                'source': filename,
                                'index': image_count,
                                'format': ext.upper(),
                                'alt_text': alt_text if alt_text else None,
                                'src': img_tag.get('src', ''),
                            }
                        }
                        
                        images.append(image_obj)
                        image_count += 1
                    
                    except Exception as e:
                        logger.warning(f"HTML image processing failed: {e}")
        
        return images
    
    async def _extract_html_images_sync(
        self,
        soup: BeautifulSoup,
        filename: str,
        base_url: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Sync image extraction (fallback when aiohttp not available)"""
        import requests
        from urllib.parse import urljoin
        
        images = []
        image_count = 0
        
        for img_tag in soup.find_all('img'):
            img_src = img_tag.get('src')
            if not img_src:
                continue
            
            img_data = None
            
            try:
                # Handle different image sources
                if img_src.startswith("data:image"):
                    # Base64 embedded
                    header, img_base64_data = img_src.split(",", 1)
                    img_data = base64.b64decode(img_base64_data)
                
                elif img_src.startswith(("http://", "https://")):
                    # Absolute URL
                    response = requests.get(img_src, timeout=10)
                    if response.status_code == 200:
                        img_data = response.content
                
                elif base_url:
                    # Relative URL
                    img_url = urljoin(base_url, img_src)
                    response = requests.get(img_url, timeout=10)
                    if response.status_code == 200:
                        img_data = response.content
                
                if not img_data:
                    continue
                
                # Determine extension
                ext = "png"
                if 'jpg' in img_src or 'jpeg' in img_src:
                    ext = 'jpg'
                elif 'gif' in img_src:
                    ext = 'gif'
                
                # Save image
                img_filename = f"{Path(filename).stem}_image_{image_count}.{ext}"
                img_path = self.figures_dir / img_filename
                
                with open(img_path, 'wb') as f:
                    f.write(img_data)
                
                img_base64 = base64.b64encode(img_data).decode()
                
                # Get alt text
                alt_text = img_tag.get('alt', '').strip()
                
                image_obj = {
                    'id': f"image_{image_count}",
                    'path': str(img_path),
                    'base64': img_base64,
                    'metadata': {
                        'source': filename,
                        'index': image_count,
                        'format': ext.upper(),
                        'alt_text': alt_text if alt_text else None,
                        'src': img_src,
                    }
                }
                
                images.append(image_obj)
                image_count += 1
            
            except Exception as e:
                logger.debug(f"Failed to extract HTML image: {e}")
        
        return images

    # ====================== Text Parser ======================
    
    async def _parse_text(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Parse plain text and markdown files"""
        logger.info(f"📃 Parsing text: {filename}")
        
        try:
            # Decode text
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('latin-1', errors='ignore')
            
            # Chunk text
            text_chunks = await self._chunk_text_simple(
                text,
                source_metadata={'filename': filename, 'doc_type': 'text'}
            )
            
            # Extract markdown tables if applicable
            tables = []
            if filename.endswith(('.md', '.markdown')):
                pattern = r'(\|.+\|[\r\n]+\|[\s\-:]+\|[\r\n]+(?:\|.+\|[\r\n]+)*)'
                for idx, match in enumerate(re.finditer(pattern, text, re.MULTILINE)):
                    table_content = match.group(1).strip()
                    tables.append({
                        'id': f"table_{idx}",
                        'content': table_content,
                        'metadata': {
                            'source': filename,
                            'table_index': idx,
                            'format': 'markdown',
                        },
                        'summary': self._generate_simple_table_summary(
                            [table_content.split('\n')], []
                        )
                    })
                
                if tables:
                    logger.info(f"Extracted {len(tables)} markdown tables")
        
        except Exception as e:
            logger.error(f"❌ Text parsing failed: {e}")
            raise
        
        return {
            'text_chunks': text_chunks,
            'tables': tables,
            'images': [],
            'document_structure': {}
        }

    # ====================== Image Parser ======================
    
    async def _parse_image(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Parse image with OCR"""
        logger.info(f"🖼️ Parsing image: {filename}")
        
        text_chunks = []
        images = []
        
        try:
            img = Image.open(io.BytesIO(content))
            
            # Save image
            img_filename = Path(filename).name
            img_path = self.figures_dir / img_filename
            img.save(img_path)
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format=img.format or "PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # OCR extraction
            ocr_text = ""
            if TESSERACT_AVAILABLE:
                try:
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        text_chunks = await self._chunk_text_simple(
                            ocr_text,
                            source_metadata={
                                'filename': filename,
                                'doc_type': 'image',
                                'source': 'ocr'
                            }
                        )
                        logger.info(f"✅ OCR extracted {len(text_chunks)} text chunks")
                except Exception as e:
                    logger.warning(f"OCR failed: {e}")
            
            image_obj = {
                'id': 'image_0',
                'path': str(img_path),
                'base64': img_base64,
                'metadata': {
                    'source': filename,
                    'format': img.format or 'PNG',
                    'size': img.size,
                    'mode': img.mode,
                    'ocr_text': ocr_text.strip() if ocr_text else None,
                }
            }
            
            # Optional: Generate description
            if self.enable_image_description:
                image_obj['description'] = await self._generate_image_description(img)
            
            images.append(image_obj)
        
        except Exception as e:
            logger.error(f"❌ Image parsing failed: {e}")
            raise
        
        return {
            'text_chunks': text_chunks,
            'tables': [],
            'images': images,
            'document_structure': {}
        }

    # ====================== Chunking Methods ======================
    
    async def _chunk_text_with_pages(
        self,
        page_texts: List[Dict[str, Any]],
        filename: str,
        total_pages: int
    ) -> List[Dict[str, Any]]:
        """Chunk text while preserving page information"""
        all_chunks = []
        chunk_id = 0
        
        for page_data in page_texts:
            page_num = page_data['page']
            page_text = page_data['text']
            
            # Chunk this page's text
            if self.enable_semantic_chunking:
                chunks = self.semantic_splitter.split_text(page_text)
            else:
                chunks = self._basic_chunk_text(page_text)
            
            # Add metadata to each chunk
            for chunk_text in chunks:
                chunk_obj = {
                    'chunk_id': chunk_id,
                    'content': chunk_text.strip(),
                    'metadata': {
                        'page': page_num,
                        'total_pages': total_pages,
                        'char_count': len(chunk_text),
                        'source': filename,
                    }
                }
                all_chunks.append(chunk_obj)
                chunk_id += 1
        
        return all_chunks
    
    async def _chunk_structured_content(
        self,
        structured_content: List[Dict[str, Any]],
        filename: str
    ) -> List[Dict[str, Any]]:
        """Chunk text while preserving section/heading context"""
        all_chunks = []
        chunk_id = 0
        
        for section in structured_content:
            heading = section.get('heading')
            level = section.get('level', 0)
            content = '\n\n'.join(section.get('content', []))
            
            if not content.strip():
                continue
            
            # Chunk section content
            if self.enable_semantic_chunking:
                chunks = self.semantic_splitter.split_text(content)
            else:
                chunks = self._basic_chunk_text(content)
            
            # Add section context to metadata
            for chunk_text in chunks:
                chunk_obj = {
                    'chunk_id': chunk_id,
                    'content': chunk_text.strip(),
                    'metadata': {
                        'section': heading,
                        'heading_level': level,
                        'char_count': len(chunk_text),
                        'source': filename,
                    }
                }
                all_chunks.append(chunk_obj)
                chunk_id += 1
        
        return all_chunks
    
    async def _chunk_text_simple(
        self,
        text: str,
        source_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Simple text chunking with metadata"""
        if not text or not text.strip():
            return []
        
        # Use semantic chunking if available
        if self.enable_semantic_chunking:
            chunks = self.semantic_splitter.split_text(text)
        else:
            chunks = self._basic_chunk_text(text)
        
        # Add metadata
        chunk_objects = []
        for chunk_id, chunk_text in enumerate(chunks):
            chunk_obj = {
                'chunk_id': chunk_id,
                'content': chunk_text.strip(),
                'metadata': {
                    'char_count': len(chunk_text),
                    **source_metadata
                }
            }
            chunk_objects.append(chunk_obj)
        
        return chunk_objects
    
    def _basic_chunk_text(self, text: str) -> List[str]:
        """Basic text chunking (fallback when LangChain not available)"""
        if not text or not text.strip():
            return []
        
        chunks = []
        start = 0
        length = len(text)
        
        while start < length:
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < length:
                # Look for sentence boundaries
                boundaries = [
                    chunk.rfind('. '),
                    chunk.rfind('.\n'),
                    chunk.rfind('? '),
                    chunk.rfind('! '),
                    chunk.rfind('\n\n'),
                ]
                boundary = max(boundaries)
                
                # Only break at boundary if it's in the second half
                if boundary > self.chunk_size // 2:
                    chunk = chunk[:boundary + 1]
                    end = start + boundary + 1
            
            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
        
        return chunks

    # ====================== Helper Methods ======================
    
    def _enrich_chunks_metadata(
        self,
        chunks: List[Dict[str, Any]],
        doc_id: str,
        filename: str,
        doc_type: str
    ) -> List[Dict[str, Any]]:
        """Add document-level metadata to all chunks"""
        for chunk in chunks:
            # Generate unique chunk ID with UUID to avoid collisions
            unique_id = str(uuid.uuid4())[:8]
            chunk['chunk_id'] = f"{doc_id}_chunk_{chunk['chunk_id']}_{unique_id}"
            
            # Add document metadata
            chunk['metadata'].update({
                'doc_id': doc_id,
                'filename': filename,
                'doc_type': doc_type,
            })
        
        return chunks
    
    def _generate_doc_id(self, content: bytes, filename: str) -> str:
        """Generate unique document ID from content hash"""
        content_hash = hashlib.md5(content).hexdigest()[:12]
        filename_clean = re.sub(r'[^a-zA-Z0-9]', '_', Path(filename).stem)
        return f"{filename_clean}_{content_hash}"
    
    def _extract_pdf_outline(self, outline: Any, level: int = 0) -> List[Dict[str, Any]]:
        """Recursively extract PDF outline/bookmarks"""
        result = []
        
        if isinstance(outline, list):
            for item in outline:
                result.extend(self._extract_pdf_outline(item, level))
        elif hasattr(outline, 'title'):
            result.append({
                'title': outline.title,
                'level': level,
            })
            if hasattr(outline, '__iter__'):
                try:
                    for child in outline:
                        result.extend(self._extract_pdf_outline(child, level + 1))
                except:
                    pass
        
        return result
    
    def _generate_simple_table_summary(
        self,
        table_data: List[List[str]],
        headers: List[str]
    ) -> str:
        """Generate simple table summary (no LLM)"""
        row_count = len(table_data) - 1 if headers else len(table_data)
        col_count = len(headers) if headers else (len(table_data[0]) if table_data else 0)
        
        summary = f"Table: {row_count} rows × {col_count} columns"
        
        if headers:
            summary += f". Columns: {', '.join(headers[:5])}"
            if len(headers) > 5:
                summary += f" and {len(headers) - 5} more"
        
        return summary
    
    async def _generate_table_summary(
        self,
        table_data: List[List[str]],
        headers: List[str]
    ) -> str:
        """Generate table summary using LLM (expensive!)"""
        # Fallback to simple summary if LLM not available
        if not self.llm_adapter:
            return self._generate_simple_table_summary(table_data, headers)
        
        try:
            # Format table for LLM
            table_str = tabulate(
                table_data[1:] if headers else table_data,
                headers=headers if headers else [],
                tablefmt="pipe"
            )
            
            # Limit table size to avoid token overflow
            if len(table_str) > 2000:
                table_str = table_str[:2000] + "..."
            
            prompt = f"""Summarize this table in 1-2 sentences (max 100 words):

{table_str}

Summary:"""
            
            # Initialize LLM if needed
            if not hasattr(self.llm_adapter, 'is_initialized') or not self.llm_adapter.is_initialized:
                await self.llm_adapter.initialize()
            
            summary = await self.llm_adapter.generate_text(prompt, max_tokens=150, temperature=0.3)
            return summary.strip()
        
        except Exception as e:
            logger.warning(f"LLM table summarization failed: {e}")
            return self._generate_simple_table_summary(table_data, headers)
    
    async def _generate_image_description(self, img: Image.Image) -> str:
        """Generate image description using multimodal LLM"""
        # Fallback to simple description if MLLM not available
        if not self.mllm_adapter:
            return f"Image: {img.size[0]}×{img.size[1]} pixels, {img.mode} mode"
        
        try:
            # Convert image to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Initialize MLLM if needed
            if not hasattr(self.mllm_adapter, 'is_initialized') or not self.mllm_adapter.is_initialized:
                await self.mllm_adapter.initialize()
            
            # Generate description
            description = await self.mllm_adapter.summarize(
                content="Describe this image in detail (1-2 sentences)",
                content_type="image",
                max_length=200,
                image_base64=img_base64
            )
            
            return description.strip()
        
        except Exception as e:
            logger.warning(f"MLLM image description failed: {e}")
            return f"Image: {img.size[0]}×{img.size[1]} pixels"
    
    # ====================== Cleanup ======================
    
    def cleanup_old_figures(self, days: int = 7) -> int:
        """Clean up old figure files (sync)"""
        import time
        
        cutoff = time.time() - (days * 86400)
        deleted = 0
        
        try:
            for file_path in self.figures_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff:
                    try:
                        file_path.unlink()
                        deleted += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
            
            if deleted > 0:
                logger.info(f"🧹 Cleaned up {deleted} old figure files")
        
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
        
        return deleted
    
    async def cleanup_async(self, days: int = 7) -> int:
        """Async version of cleanup"""
        return await asyncio.to_thread(self.cleanup_old_figures, days)
