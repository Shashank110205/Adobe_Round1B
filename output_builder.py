# output_builder.py

import json

def build_output(input_data, ranked_chunks, timestamp):
    extracted_sections = []
    subsection_analysis = []

    for i, chunk in enumerate(ranked_chunks):
        extracted_sections.append({
            "document": chunk["document"],
            "section_title": chunk["section_title"],
            "importance_rank": i + 1,
            "page_number": chunk["page_number"]
        })
        subsection_analysis.append({
            "document": chunk["document"],
            "refined_text": chunk["text"],
            "page_number": chunk["page_number"]
        })

    output = {
        "metadata": {
            "input_documents": [doc["filename"] for doc in input_data["documents"]],
            "persona": input_data["persona"]["role"],
            "job_to_be_done": input_data["job_to_be_done"]["task"],
            "processing_timestamp": timestamp
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    with open("challenge1b_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
