# main1.py

import json
import os
import re
from datetime import datetime, timezone
from nltk.tokenize import sent_tokenize
# import nltk
# nltk.download('punkt')

from model_utils_e5 import get_embedding_e5, cosine_similarity
from pdf_utils import extract_chunks_by_headings
from scoring_utils import (
    extract_keywords,
    keyword_match_score,
    compute_final_score
)
from output_builder import build_output

# Load input
with open("input.json", "r", encoding="utf-8") as f:
    input_data = json.load(f)

persona = input_data["persona"]["role"]
job = input_data["job_to_be_done"]["task"]
documents = input_data["documents"]
pdf_folder = "pdf"

# Prepare query
query_instruction = "query:"
doc_instruction = "passage:"
query_text = f"{persona}. Task: {job}"
query_keywords = extract_keywords(query_text)
query_embedding = get_embedding_e5(query_text, instruction=query_instruction)

all_chunks = []

# Function to refine text
def refine_text(text):
    sentences = sent_tokenize(text)

    meaningful = []
    for s in sentences:
        cleaned = s.strip("•*-• \n\t")  # remove bullets or extra symbols
        if len(cleaned) > 30 and not cleaned.startswith(("•", "-", "*")):
            meaningful.append(cleaned)
        if len(meaningful) == 3:
            break

    return " ".join(meaningful) if meaningful else text.strip()


# Score and store chunks
for doc in documents:
    filename = doc["filename"]
    filepath = os.path.join(pdf_folder, filename)

    print(f"🔍 Processing: {filename}")
    chunks = extract_chunks_by_headings(filepath)

    for chunk in chunks:
        chunk_text = chunk["text"]
        if isinstance(chunk_text, list):
            chunk_text = " ".join(chunk_text)

        refined_chunk_text = refine_text(chunk_text)

        passage_embedding = get_embedding_e5(refined_chunk_text, instruction=doc_instruction)
        similarity = cosine_similarity(query_embedding, passage_embedding)
        match_score = keyword_match_score(refined_chunk_text, query_keywords)
        final_score = compute_final_score(similarity, match_score)

        all_chunks.append({
            "document": filename,
            "section_title": chunk["section_title"],
            "text": chunk_text,  # original for context
            "refined_text": refined_chunk_text,  # for output
            "page_number": chunk["page_number"],
            "similarity_score": similarity,
            "match_score": match_score,
            "final_score": final_score
        })

# Filter out generic or malformed section titles
filtered_chunks = [
    c for c in all_chunks
    if len(c["section_title"].split()) > 3
    and "untitled" not in c["section_title"].lower()
    and not re.match(r"(?i)^(introduction|conclusion)$", c["section_title"].strip())
]

# Pick top 5 diverse chunks
seen_docs = set()
diverse_chunks = []
for chunk in sorted(filtered_chunks, key=lambda x: x["final_score"], reverse=True):
    if chunk["document"] not in seen_docs:
        diverse_chunks.append(chunk)
        seen_docs.add(chunk["document"])
    if len(diverse_chunks) == 5:
        break

# Write final output
timestamp = datetime.now(timezone.utc).isoformat()
build_output(input_data, diverse_chunks, timestamp)

print("✅ Output written to challenge1b_output.json")
