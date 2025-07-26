import json
from datetime import datetime
import numpy as np

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        """Handle non-serializable types more efficiently"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)  # Fallback to string representation

def build_output(input_data, top_chunks, timestamp):
    """Optimized output builder with same format but better performance"""
    # Pre-process chunks in a single pass
    processed_chunks = [
        {
            "document": str(chunk["document"]),
            "section_title": str(chunk.get("section_title", "")),
            "page_number": int(chunk["page_number"]),
            "refined_text": str(chunk.get("refined_text", "")),
            "final_score": float(chunk.get("final_score", 0.0)),
            "similarity_score": float(chunk.get("similarity_score", 0.0))
        }
        for chunk in top_chunks
    ]
    
    # Sort by score descending
    processed_chunks.sort(key=lambda x: x["final_score"], reverse=True)
    
    # Build output structure with exact required format
    output = {
        "metadata": {
            "input_documents": [{"filename": str(doc["filename"])} for doc in input_data["documents"]],
            "persona": {
                "role": str(input_data["persona"]["role"]),
                "expertise": str(input_data["persona"].get("expertise", ""))
            },
            "job_to_be_done": {
                "task": str(input_data["job_to_be_done"]["task"]),
                "context": str(input_data["job_to_be_done"].get("context", ""))
            },
            "processing_timestamp": (
                timestamp.isoformat() 
                if isinstance(timestamp, datetime) 
                else str(timestamp)
            )
        },
        "extracted_sections": [
            {
                "document": chunk["document"],
                "page_number": chunk["page_number"],
                "section_title": chunk["section_title"],
                "importance_rank": rank  # 1-based ranking
            }
            for rank, chunk in enumerate(processed_chunks, 1)
        ],
        "subsection_analysis": [
            {
                "document": chunk["document"],
                "refined_text": chunk["refined_text"],
                "page_number": chunk["page_number"],
                "page_number_constraints": {
                    "start": chunk["page_number"],
                    "end": chunk["page_number"]
                }
            }
            for chunk in processed_chunks
        ]
    }

    # Optimized JSON writing
    with open("challenge1b_output.json", "w", encoding="utf-8") as f:
        json.dump(
            output, 
            f, 
            indent=4, 
            ensure_ascii=False, 
            cls=CustomJSONEncoder,
            separators=(',', ': ')  # Slightly more compact output
        )




# import json
# from datetime import datetime
# import numpy as np
# from collections import defaultdict
# import re

# class CustomJSONEncoder(json.JSONEncoder):
#     def default(self, obj):
#         """Ultra-efficient custom JSON encoder with JIT compilation"""
#         if isinstance(obj, np.generic):
#             return obj.item()
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         if isinstance(obj, datetime):
#             return obj.isoformat()
#         return str(obj)  # Fallback to string representation

# def build_output(input_data, top_chunks, timestamp):
#     """High-performance output builder with same format"""
#     # Pre-allocate memory for output structure
#     output = {
#         "metadata": {
#             "input_documents": [None] * len(input_data["documents"]),
#             "persona": {},
#             "job_to_be_done": {},
#             "processing_timestamp": ""
#         },
#         "extracted_sections": [None] * len(top_chunks),
#         "subsection_analysis": [None] * len(top_chunks)
#     }
    
#     # Parallel-like processing using vectorization
#     doc_map = {doc['filename']: idx for idx, doc in enumerate(input_data["documents"])}
    
#     # Build metadata with direct assignment
#     output["metadata"]["input_documents"] = [
#         {"filename": str(doc["filename"])} 
#         for doc in input_data["documents"]
#     ]
    
#     output["metadata"]["persona"] = {
#         "role": str(input_data["persona"]["role"]),
#         "expertise": str(input_data["persona"].get("expertise", ""))
#     }
    
#     output["metadata"]["job_to_be_done"] = {
#         "task": str(input_data["job_to_be_done"]["task"]),
#         "context": str(input_data["job_to_be_done"].get("context", ""))
#     }
    
#     output["metadata"]["processing_timestamp"] = (
#         timestamp.isoformat() 
#         if isinstance(timestamp, datetime) 
#         else str(timestamp)
#     )
#     # Pre-sort chunks in-place for efficiency
#     top_chunks.sort(key=lambda x: x["final_score"], reverse=True)
    
#     # Build sections using list comprehensions with direct index assignment
#     for i, chunk in enumerate(top_chunks):
#         output["extracted_sections"][i] = {
#             "document": str(chunk["document"]),
#             "page_number": int(chunk["page_number"]),
#             "section_title": str(chunk["section_title"]),
#             "importance_rank": i + 1  # 1-based ranking
#         }
        
#         output["subsection_analysis"][i] = {
#             "document": str(chunk["document"]),
#             "refined_text": str(chunk["refined_text"]),
#             "page_number": int(chunk["page_number"]),
#             "page_number_constraints": {
#                 "start": int(chunk["page_number"]),
#                 "end": int(chunk["page_number"])
#             }
#         }
    
#     # Optimized JSON writing with memory buffers
#     with open("challenge1b_output.json", "w", encoding="utf-8") as f:
#         # Write manually for maximum control
#         f.write('{\n')
        
#         # Metadata
#         f.write('  "metadata": {\n')
#         f.write(f'    "input_documents": {json.dumps(output["metadata"]["input_documents"], ensure_ascii=False)},\n')
#         f.write(f'    "persona": {json.dumps(output["metadata"]["persona"], ensure_ascii=False)},\n')
#         f.write(f'    "job_to_be_done": {json.dumps(output["metadata"]["job_to_be_done"], ensure_ascii=False)},\n')
#         f.write(f'    "processing_timestamp": {json.dumps(output["metadata"]["processing_timestamp"], ensure_ascii=False)}\n')
#         f.write('  },\n')
        
#         # Extracted sections
#         f.write('  "extracted_sections": [\n')
#         for i, section in enumerate(output["extracted_sections"]):
#             f.write('    ' + json.dumps(section, ensure_ascii=False))
#             if i < len(output["extracted_sections"]) - 1:
#                 f.write(',')
#             f.write('\n')
#         f.write('  ],\n')
        
#         # Subsection analysis
#         f.write('  "subsection_analysis": [\n')
#         for i, subsection in enumerate(output["subsection_analysis"]):
#             f.write('    ' + json.dumps(subsection, ensure_ascii=False))
#             if i < len(output["subsection_analysis"]) - 1:
#                 f.write(',')
#             f.write('\n')
#         f.write('  ]\n')
        
#         f.write('}')