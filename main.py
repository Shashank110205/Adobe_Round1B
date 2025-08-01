import json
import os
import time
import re
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

# Configuration with environment variable support
MAX_DOCUMENTS = int(os.getenv('MAX_DOCUMENTS', 10))
MAX_TOP_CHUNKS = int(os.getenv('MAX_TOP_CHUNKS', 5))
MAX_THREADS = int(os.getenv('MAX_THREADS', 4))
MIN_CHUNK_LENGTH = int(os.getenv('MIN_CHUNK_LENGTH', 100))

# Directory paths from environment variables with defaults
INPUT_DIR = os.getenv('INPUT_DIR', './input')
PDF_FOLDER = os.path.join(INPUT_DIR, 'PDFs')

# Input and output file paths (both in INPUT_DIR now)
INPUT_FILE = os.path.join(INPUT_DIR, 'challenge1b_input.json')
OUTPUT_FILE = os.path.join(INPUT_DIR, 'challenge1b_output.json')

def ensure_directories():
    """Ensure input directory and PDF subfolder exist"""
    directories = [INPUT_DIR, PDF_FOLDER]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Created directory: {directory}")

def validate_input_files():
    """Validate that required input files exist"""
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        print(f"   Please ensure input.json is mounted to {INPUT_DIR}")
        return False
    
    if not os.path.exists(PDF_FOLDER):
        print(f"❌ PDF folder not found: {PDF_FOLDER}")
        print(f"   Please ensure PDF files are mounted to {PDF_FOLDER}")
        return False
    
    # Check if PDF folder has any PDF files
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"⚠️  No PDF files found in {PDF_FOLDER}")
        print(f"   Available files: {os.listdir(PDF_FOLDER) if os.path.exists(PDF_FOLDER) else 'Directory does not exist'}")
    
    return True

def process_document(doc: dict) -> list:
    """Process a single document with enhanced PDF extraction"""
    filepath = os.path.join(PDF_FOLDER, doc["filename"])
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"⚠️ File not found: {filepath}")
        return []
    
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
    print("🚀 Starting PDF Processing Pipeline...")
    print(f"📂 Working Directory: {INPUT_DIR}")
    print(f"📄 PDF Directory: {PDF_FOLDER}")
    print(f"📝 Output will be saved to: {OUTPUT_FILE}")
    
    start_time = time.time()
    
    # Ensure directories exist
    ensure_directories()
    
    # Validate input files
    if not validate_input_files():
        print("❌ Input validation failed. Exiting...")
        return
    
    # Load input
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            input_data = json.load(f)
        print(f"✅ Successfully loaded input from: {INPUT_FILE}")
    except Exception as e:
        print(f"❌ Error loading input file: {str(e)}")
        return
    
    persona = input_data["persona"]["role"]
    job = input_data["job_to_be_done"]["task"]
    documents = input_data["documents"][:MAX_DOCUMENTS]
    
    print(f"👤 Persona: {persona}")
    print(f"🎯 Task: {job}")
    print(f"📚 Documents to process: {len(documents)}")
    
    # Initialize embedding generator
    try:
        embedder = EmbeddingGenerator()
        print("✅ Embedding generator initialized")
    except Exception as e:
        print(f"❌ Error initializing embedder: {str(e)}")
        return
    
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
        print(f"   Available files in {PDF_FOLDER}: {os.listdir(PDF_FOLDER) if os.path.exists(PDF_FOLDER) else 'Directory does not exist'}")
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
    
    # Write output to the same directory as input
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"✅ Output successfully saved to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ Error saving output: {str(e)}")
        return
    
    # Performance metrics
    proc_time = time.time() - start_time
    print(f"\n✅ Processing completed in {proc_time:.2f} seconds")
    print(f"📊 Processed {len(documents)} documents, {len(all_chunks)} sections")
    print(f"🎯 Selected {len(selected_chunks)} most relevant sections")
    print(f"💾 Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
