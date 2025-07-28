# 📘 Approach Explanation: Persona-Aware Document Intelligence System

## 🧠 Overview

This system is designed to intelligently extract the most relevant sections from a set of PDF documents based on a user’s *persona* and a *job to be done*. It leverages ONNX-based sentence embeddings, contextual keyword extraction, semantic similarity, and PDF parsing to return ranked, refined, and meaningful document segments with rich metadata.

---

## 🔧 Key Components

### 1. **Embedding Generator**
- File: `embedding_generator.py`
- Purpose: Generates semantic vector embeddings using a pre-trained ONNX version of the `e5-small-v2` model.
- Features:
  - Uses `onnxruntime` for fast inference.
  - Tokenization via `tokenizers` library.
  - Mean pooling + cosine normalization for embedding representation.
  - Caching support for performance.
  - Supports both single and batch embedding generation.

---

### 2. **PDF Chunk Extractor**
- File: `pdf_processor.py`
- Purpose: Converts PDF pages into structured text chunks using **PyMuPDF** (`fitz`).
- Strategy:
  - Detects headings and sections based on font size, boldness, center alignment, and vertical spacing.
  - Aggregates content between detected headings into logical "chunks".
  - Returns cleaned, page-wise sections with detected titles and body text.

---

### 3. **Text Utilities**
- File: `text_utils.py`
- Includes key utilities:
  
  #### a. `refine_text(text)`
  - Extracts readable, natural language sentences from raw chunk text.
  - Uses punctuation and quote-awareness to prevent sentence fragmentation.

  #### b. `extract_keywords(text)`
  - Identifies top-n frequent, meaningful words from the text.
  
  #### c. `generate_dynamic_keywords(persona, job, embedder)`
  - Uses semantic similarity to expand base keywords using a **universal vocabulary**.
  - Embeds both base and vocabulary words, filters based on cosine similarity.

  #### d. `generate_contextual_title(text, query_keywords, original_title, embedder)`
  - Automatically generates section titles when original ones are missing or uninformative.
  - Uses semantic matching of phrases to the query intent.

  #### e. `calculate_cosine_similarity(vec1, vec2)`
  - Robust cosine similarity computation with zero-div checks.

---

### 4. **Output Builder**
- File: `output_builder.py`
- Structures the final output into a JSON format with:
  - Metadata (persona, job, timestamps)
  - Extracted section titles
  - Refined section content for downstream applications
- Output: `challenge1b_output.json`

---

## ⚙️ Main Pipeline: `main()`

### Step-by-Step Flow:

#### ✅ 1. **Input Parsing**
- Reads `input.json`:
  ```json
  {
    "persona": {"role": "Travel Planner"},
    "job_to_be_done": {"task": "Plan a 5-day tour in Italy"},
    "documents": [{"filename": "italy_travel.pdf"}, ...]
  }
  ```

#### 🧠 2. **Query Embedding Generation**
- Combines persona and task into a semantic query.
- Extracts and expands keywords using `generate_dynamic_keywords()`.

#### 📄 3. **Document Chunking**
- Extracts and structures content from up to `MAX_DOCUMENTS` PDF files using `ThreadPoolExecutor`.

#### 🧹 4. **Preprocessing**
- Cleans each chunk.
- Refines text.
- Scores keyword matches for each chunk.

#### 🔍 5. **Semantic Scoring**
- Embeds all relevant text chunks using `get_embeddings_batch()`.
- Calculates similarity with the query embedding.
- Combines:
  - Cosine similarity
  - Keyword match score
  - Dynamic keyword boosting

#### 🏆 6. **Top Chunk Selection**
- Sorts all chunks by `final_score`.
- Selects up to `MAX_TOP_CHUNKS`, ensuring **document diversity** (max 2 per file).
- Uses `generate_contextual_title()` to assign informative titles.

#### 📝 7. **Output Generation**
- Formats everything into a standardized output structure.
- Writes final result to `challenge1b_output.json`.

---

## 🔑 Techniques Used

| Technique                        | Purpose                                     |
|----------------------------------|---------------------------------------------|
| ONNX Inference with `e5-small-v2`| Fast, accurate sentence embeddings          |
| Mean Pooling + Normalization     | Stable sentence representations             |
| PDF Section Detection            | Structure-aware content segmentation        |
| Cosine Similarity                | Semantic comparison                         |
| Dynamic Keyword Expansion        | Context-aware matching                      |
| Title Regeneration               | Makes content self-explanatory              |
| Multithreading (`ThreadPoolExecutor`) | Efficient document parsing              |

---

## 📁 Folder Structure (Simplified)

```
project/
│
├── main.py
├── input.json
├── challenge1b_output.json
│
├── optimized_util/
│   ├── embedding_generator.py
│   ├── pdf_processor.py
│   ├── text_utils.py
│   └── output_builder.py
│
├── model/
│   └── e5-small-v2-onnx/
│       ├── model.onnx
│       └── tokenizer/tokenizer.json
│
└── pdf/
    └── *.pdf
```

---

## ✅ Final Output Format

Sample schema from `challenge1b_output.json`:
```json
{
  "metadata": { ... },
  "extracted_sections": [
    {
      "document": "italy_travel.pdf",
      "page_number": 5,
      "section_title": "Day 1 Florence Tour",
      "importance_rank": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "italy_travel.pdf",
      "refined_text": "Begin your journey in Florence with a guided tour...",
      "page_number": 5,
      "page_number_constraints": {
        "start": 5,
        "end": 5
      }
    }
  ]
}
```

---

## 🧪 Performance and Limitations

| Metric                   | Value             |
|--------------------------|-------------------|
| Avg Processing Time      | ~10–15 seconds (depending on file size) |
| Max PDF Size             | 10 PDFs            |
| Max Extracted Chunks     | 5 high-quality sections |
| Limitation               | Heading detection might miss in scanned/non-standard PDFs |

---

## 📌 Suggestions for Future Improvements

- Add OCR support for scanned documents (e.g., using Tesseract).
- Visual PDF preview with highlighted segments.
- UI for persona/job inputs and result exploration.
- GPU inference for faster embedding generation.