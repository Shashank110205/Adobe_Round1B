# adobe_round1b/pdf_utils.py

import fitz  # PyMuPDF
import re

def extract_chunks_by_headings(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    
    # Configuration
    MIN_HEADING_WORDS = 2  # Reduced from 4
    MAX_HEADING_WORDS = 12
    MIN_CHUNK_LENGTH = 50  # characters
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        
        # Improved heading detection
        font_sizes = []
        bold_fonts = set()
        for block in blocks:
            if block["type"] == 0:  # text only
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_sizes.append(span["size"])
                        if "bold" in span["font"].lower():
                            bold_fonts.add(span["font"].lower())
        
        avg_font = sum(font_sizes)/len(font_sizes) if font_sizes else 11
        heading_threshold = avg_font * 1.2  # More dynamic threshold
        
        current_chunk = []
        current_title = "Introduction"  # Default title
        
        for block in blocks:
            if block["type"] != 0:
                continue  # Skip non-text blocks
                
            for line in block["lines"]:
                line_text = " ".join(span["text"].strip() for span in line["spans"])
                if not line_text:
                    continue
                    
                # Enhanced heading detection
                is_heading = False
                for span in line["spans"]:
                    if (span["size"] > heading_threshold or 
                        span["font"].lower() in bold_fonts):
                        is_heading = True
                        break
                
                if is_heading:
                    # Save previous chunk if valid
                    if current_chunk and len(" ".join(current_chunk)) > MIN_CHUNK_LENGTH:
                        chunks.append({
                            "section_title": current_title,
                            "text": " ".join(current_chunk),
                            "page_number": page_num + 1
                        })
                    
                    # Process new heading
                    words = line_text.split()
                    if (MIN_HEADING_WORDS <= len(words) <= MAX_HEADING_WORDS and
                        not line_text.lower().startswith(("page", "footer", "header"))):
                        current_title = line_text
                    current_chunk = []
                else:
                    current_chunk.append(line_text)
        
        # Add final chunk of page
        if current_chunk and len(" ".join(current_chunk)) > MIN_CHUNK_LENGTH:
            chunks.append({
                "section_title": current_title,
                "text": " ".join(current_chunk),
                "page_number": page_num + 1
            })
    
    return chunks