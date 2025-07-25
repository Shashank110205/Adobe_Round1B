# main.py

import json
import os
from model_utils_bge import get_embedding_bge, cosine_similarity
from pdf_utils import extract_chunks_from_pdf
from scoring_utils import (
    extract_keywords,
    keyword_match_score,
    compute_final_score
)
from output_builder import build_output
from datetime import datetime, timezone

# 1. Load input JSON
with open("input.json", "r", encoding="utf-8") as f:
    input_data = json.load(f)

persona = input_data["persona"]["role"]
job = input_data["job_to_be_done"]["task"]
documents = input_data["documents"]
pdf_folder = "pdf"
instruction_query = f"Represent this planning task for embedding: {job}, from the viewpoint of a {persona}"
instruction_doc = f"Represent this content for answering the planning task: {job}, as a {persona}"

query = f"{persona}. Task: {job}"
query_keywords = extract_keywords(query)
query_embedding = get_embedding_bge(query, instruction=instruction_query)

# 2. Process each document
all_chunks = []

for doc in documents:
    filename = doc["filename"]
    filepath = os.path.join(pdf_folder, filename)

    print(f"🔍 Processing: {filename}")
    chunks = extract_chunks_from_pdf(filepath)

    for chunk in chunks:
        text = chunk["text"]
        emb = get_embedding_bge(text, instruction=instruction_doc)
        sim = cosine_similarity(query_embedding, emb)
        kw_score = keyword_match_score(text, query_keywords)
        final_score = compute_final_score(sim, kw_score)

        chunk["final_score"] = final_score
        chunk["document"] = chunk["document"].split("/")[-1].split("\\")[-1]
        all_chunks.append(chunk)

# 3. Sort top sections
top_chunks = sorted(all_chunks, key=lambda x: x["final_score"], reverse=True)[:5]

# 4. Build output JSON
timestamp = datetime.now(timezone.utc).isoformat()
build_output(input_data, top_chunks, timestamp)

print("✅ Output written to challenge1b_output.json")
