import fitz  # PyMuPDF
import re

def extract_chunks(pdf_path: str) -> list:
    """Universal PDF text extraction for any document type"""
    doc = fitz.open(pdf_path)
    chunks = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)["blocks"]
        current_chunk = []
        current_title = ""
        prev_y = 0
        
        for block in blocks:
            if block["type"] != 0:
                continue  # Skip non-text blocks
                
            for line in block["lines"]:
                line_text = "".join(span["text"] for span in line["spans"]).strip()
                if not line_text:
                    continue
                
                # Heading detection - universal approach
                is_heading = False
                if line["spans"]:
                    # Check font characteristics
                    span = line["spans"][0]
                    is_large = span["size"] > 12
                    is_bold = span["flags"] & 2  # Bold flag
                    is_centered = abs(span["origin"][0] - (page.rect.width / 2)) < 50
                    
                    # Check text characteristics
                    is_short = len(line_text.split()) <= 8
                    is_title_case = line_text.istitle() or line_text.isupper()
                    no_ending_punct = not line_text.endswith(('.', '!', '?', ':'))
                    
                    # Combined heading detection
                    if (is_large or is_bold or is_centered) and (is_short and is_title_case and no_ending_punct):
                        is_heading = True
                
                # Position-based section detection
                current_y = line["spans"][0]["origin"][1] if line["spans"] else prev_y + 20
                is_new_section = current_y - prev_y > 50
                prev_y = current_y
                
                if is_heading or is_new_section:
                    # Save current chunk
                    if current_chunk:
                        chunk_text = " ".join(current_chunk)
                        if chunk_text.strip() and len(chunk_text) > 50:
                            chunks.append({
                                "section_title": current_title,
                                "text": chunk_text,
                                "page_number": page_num + 1
                            })
                        current_chunk = []
                    
                    # Set new title
                    if is_heading:
                        current_title = line_text
                
                current_chunk.append(line_text)
        
        # Save last chunk of page
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if chunk_text.strip() and len(chunk_text) > 50:
                chunks.append({
                    "section_title": current_title,
                    "text": chunk_text,
                    "page_number": page_num + 1
                })
    
    return chunks