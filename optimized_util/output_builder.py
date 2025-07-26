import json
import re
import os

def build_output(input_data: dict, top_chunks: list, timestamp: str) -> dict:
    """Build output structure with enhanced formatting"""
    # Build metadata
    metadata = {
        "input_documents": [{"filename": doc["filename"]} for doc in input_data["documents"]],
        "persona": {
            "role": input_data["persona"]["role"],
            "expertise": input_data["persona"].get("expertise", "")
        },
        "job_to_be_done": {
            "task": input_data["job_to_be_done"]["task"],
            "context": input_data["job_to_be_done"].get("context", "")
        },
        "processing_timestamp": timestamp,
        "system_version": "PersonaAI 1.3",
        "optimization_level": "high"
    }
    
    # Build sections
    extracted_sections = []
    subsection_analysis = []
    
    for i, chunk in enumerate(top_chunks):
        # Clean section title
        title = re.sub(r'[^a-zA-Z0-9 \-\',.:;!?]', '', chunk["section_title"])
        title = re.sub(r'\s+', ' ', title).strip()
        
        extracted_sections.append({
            "document": chunk["document"],
            "page_number": chunk["page_number"],
            "section_title": title[:120],  # Limit title length
            "importance_rank": i+1
        })
        
        subsection_analysis.append({
            "document": chunk["document"],
            "refined_text": chunk["refined_text"],
            "page_number": chunk["page_number"],
            "page_number_constraints": {
                "start": chunk["page_number"],
                "end": chunk["page_number"]
            }
        })
    
    return {
        "metadata": metadata,
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }