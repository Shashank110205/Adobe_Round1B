from title_utils import generate_title
import pdfplumber

def extract_chunks_from_pdf(filepath):
    chunks = []

    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue

            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]

            for para in paragraphs:
                title = generate_title(para)
                chunks.append({
                    "document": filepath.split("/")[-1],
                    "page_number": page_num,
                    "section_title": title,
                    "text": para
                })

    return chunks
