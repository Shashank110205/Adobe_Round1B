import json
import os
import time
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from optimized_util.embedding_generator import EmbeddingGenerator
from optimized_util.pdf_processor import extract_chunks
from optimized_util.text_utils import (
    refine_text, 
    extract_keywords, 
    keyword_match_score, 
    generate_contextual_title,
    generate_dynamic_keywords,
    calculate_cosine_similarity
)
from optimized_util.output_builder import build_output

# Configuration
MAX_DOCUMENTS = 10
MAX_TOP_CHUNKS = 5
MAX_THREADS = 4
PDF_FOLDER = "pdf"
MIN_CHUNK_LENGTH = 100

def process_document(doc: dict) -> list:
    """Process a single document with enhanced PDF extraction"""
    filepath = os.path.join(PDF_FOLDER, doc["filename"])
    try:
        chunks = extract_chunks(filepath)
        valid_chunks = []
        
        for chunk in chunks:
            # Clean and validate chunk
            chunk["text"] = re.sub(r'\s+', ' ', chunk["text"]).strip()
            if len(chunk["text"]) < MIN_CHUNK_LENGTH:
                continue
                
            # Add document info
            chunk["document"] = doc["filename"]
            valid_chunks.append(chunk)
            
        return valid_chunks
    except Exception as e:
        print(f"⚠️ Error processing {doc['filename']}: {str(e)}")
        return []

def main():
    start_time = time.time()
    
    # Load input
    with open("input.json", "r", encoding="utf-8") as f:
        input_data = json.load(f)
    
    persona = input_data["persona"]["role"]
    job = input_data["job_to_be_done"]["task"]
    documents = input_data["documents"][:MAX_DOCUMENTS]
    
    # Initialize embedding generator
    embedder = EmbeddingGenerator()
    
    # Prepare query
    query_text = f"{persona}. Task: {job}"
    query_keywords = extract_keywords(query_text)
    
    # Generate dynamic keywords based on persona and job
    dynamic_keywords = generate_dynamic_keywords(persona, job, embedder)
    print(f"🔑 Generated dynamic keywords: {', '.join(dynamic_keywords)}")
    
    # Get query embedding
    query_embedding = embedder.get_embedding(query_text, "query:").flatten()
    
    # Parallel document processing
    all_chunks = []
    print(f"📂 Processing {len(documents)} documents with {MAX_THREADS} threads...")
    
    with ThreadPoolExecutor(max_workers=min(MAX_THREADS, len(documents))) as executor:
        futures = [executor.submit(process_document, doc) for doc in documents]
        for i, future in enumerate(futures):
            chunks = future.result()
            all_chunks.extend(chunks)
            print(f"  ✅ Document {i+1}: Extracted {len(chunks)} sections")
    
    if not all_chunks:
        print("❌ No content extracted from documents. Check PDF format and extraction logic.")
        return
    
    # Precompute refined text and keyword scores
    print("🔍 Refining text and calculating keyword matches...")
    for chunk in all_chunks:
        chunk["refined_text"] = refine_text(chunk["text"])
        chunk["match_score"] = keyword_match_score(chunk["refined_text"], query_keywords)
    
    # Batch process embeddings for relevant chunks
    chunks_to_embed = [chunk for chunk in all_chunks if chunk["match_score"] > 0.1]
    
    if chunks_to_embed:
        print(f"🧠 Generating embeddings for {len(chunks_to_embed)} relevant sections...")
        texts_to_embed = [chunk["refined_text"] for chunk in chunks_to_embed]
        chunk_embeddings = embedder.get_embeddings_batch(texts_to_embed, "passage:")
        
        for chunk, embedding in zip(chunks_to_embed, chunk_embeddings):
            similarity = calculate_cosine_similarity(query_embedding, embedding)
            chunk["similarity"] = max(similarity, 0.0)
    
    # Calculate final scores
    for chunk in all_chunks:
        similarity = chunk.get("similarity", chunk["match_score"] * 0.5)
        chunk["final_score"] = (similarity * 0.7) + (chunk["match_score"] * 0.3)
        
        # Apply dynamic keyword boosting
        text = chunk["refined_text"].lower()
        matches = sum(1 for kw in dynamic_keywords if kw in text)
        if matches > 0:
            boost_factor = 1 + (0.1 * matches)  # 10% per match
            chunk["final_score"] *= min(boost_factor, 1.5)  # Cap at 50% boost
        
        # Generate contextual title
        chunk["section_title"] = generate_contextual_title(
            chunk["text"],
            query_keywords,
            chunk.get("section_title", ""),
            embedder
        )
    
    # Select top chunks with document diversity
    all_chunks.sort(key=lambda x: x["final_score"], reverse=True)
    selected_chunks = []
    doc_counts = defaultdict(int)
    
    print("🏆 Selecting top sections with document diversity...")
    for chunk in all_chunks:
        doc = chunk["document"]
        if doc_counts[doc] < 2:  # Max 2 per document
            selected_chunks.append(chunk)
            doc_counts[doc] += 1
            print(f"  ★ {doc_counts[doc]} from {doc}: {chunk['section_title'][:50]}...")
        if len(selected_chunks) >= MAX_TOP_CHUNKS:
            break
    
    # Generate output
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    output = build_output(input_data, selected_chunks, timestamp)
    
    # Write output
    with open("challenge1b_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Performance metrics
    proc_time = time.time() - start_time
    print(f"\n✅ Processing completed in {proc_time:.2f} seconds")
    print(f"📊 Processed {len(documents)} documents, {len(all_chunks)} sections")
    print(f"🎯 Selected {len(selected_chunks)} most relevant sections")
    print(f"💾 Output saved to challenge1b_output.json")

if __name__ == "__main__":
    main()