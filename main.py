import json
import os
from model_utils import get_combined_similarity_batch
from pdf_utils import extract_chunks_from_pdf
from scoring_utils import (
    extract_keywords,
    keyword_match_score,
    compute_final_score
)
from output_builder import build_output
from datetime import datetime

# 1. Load input JSON
with open("input.json", "r", encoding="utf-8") as f:
    input_data = json.load(f)

persona = input_data["persona"]["role"]
job = input_data["job_to_be_done"]["task"]
documents = input_data["documents"]
pdf_folder = "pdf"

query = f"{persona}. Task: {job}"
query_keywords = extract_keywords(query)

# 2. Process each document
all_chunks = []

for doc in documents:
    filename = doc["filename"]
    filepath = os.path.join(pdf_folder, filename)

    print(f"🔍 Processing: {filename}")
    chunks = extract_chunks_from_pdf(filepath)
    texts = [chunk["text"] for chunk in chunks]

    # ✅ Batch compute similarities
    similarities = get_combined_similarity_batch(query, texts, w_minilm=0.4, w_bge=0.6)

    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        sim = similarities[i]
        kw_score = keyword_match_score(text, query_keywords)
        final_score = compute_final_score(sim, kw_score)

        chunk["final_score"] = final_score
        chunk["document"] = chunk["document"].split("/")[-1].split("\\")[-1]
        all_chunks.append(chunk)

# 3. Sort top sections
top_chunks = sorted(all_chunks, key=lambda x: x["final_score"], reverse=True)[:5]

# 4. Build output JSON
timestamp = datetime.utcnow().isoformat()
build_output(input_data, top_chunks, timestamp)

print("✅ Output written to challenge1b_output.json")
