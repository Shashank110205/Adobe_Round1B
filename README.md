# 🧠 Adobe Hackathon Round 1B — Persona-Aware PDF Section Ranker

A solution to extract and rank the most relevant PDF sections for a given **persona** and **job-to-be-done**, using optimized ONNX-based embeddings and contextual keyword matching.

---

## 📌 Problem Statement

> Given multiple PDFs, a **persona** (e.g., travel agent, researcher) and their **task**, extract the top 5 most relevant sections across documents — and return them in a structured JSON format.

---

## Team
Name : InnovationNation 

Members : 
- [Shashank Tiwari](https://github.com/Shashank110205)
- [Badal Singh](https://github.com/Badalsingh2)
- [Vivian Ludrick](https://github.com/vivalchemy)

---

## 🚀 Features

- 🔹 **E5-Small-V2 ONNX** model for efficient, offline-compatible embeddings
- 🔹 **Dynamic keyword expansion** using a universal vocabulary
- 🔹 **Robust heading-based chunking** using font and spacing logic
- 🔹 **Caching & batching** for speed and performance
- 🔹 **Contextual refinement** of extracted text and section titles

---

## 🧠 Approach Summary

<details>
<summary>🔍 Click to expand</summary>

### 🔹 Step 1: Input Parsing
- Input JSON defines:
  - `persona`
  - `job_to_be_done`
  - List of PDFs to process

### 🔹 Step 2: Chunk Extraction
- PDFs processed using **PyMuPDF**
- Sections extracted based on:
  - Font size
  - Boldness
  - Heading heuristics

### 🔹 Step 3: Embedding Generation
- Each query and section is converted into embeddings using ONNX
- `[query:]` and `[passage:]` prefixes are used (as per E5 paper)
- Results are normalized and cached

### 🔹 Step 4: Scoring & Ranking
- Each chunk scored by:
  - **Cosine similarity** with query
  - **Keyword match score**
- Top 5 chunks selected across documents (1 per document)

### 🔹 Step 5: Output
- Final JSON includes:
  - Metadata (persona, task, timestamp)
  - Extracted sections
  - Refined content for each section

</details>

---

## 🗂️ Folder Structure

```.
├── main.py # Optimized pipeline using ONNX
├── main1.py # NLTK-based simpler baseline version
├── input.json # Input with persona, task, and documents
├── challenge1b_output.json # Final structured output
├── optimized_util/ # Optimized embedding, PDF, scoring utilities
│ ├── embedding_generator.py
│ ├── output_builder.py
│ ├── pdf_processor.py
│ └── text_utils.py
├── utils/ # Simpler utilities for main1.py
├── tokenizer/ # Tokenizer JSON for ONNX model
├── pdf/ # Folder containing input PDF files
└── README.md # This file
```

---

## 📥 Installation Steps:

1. Clone the git repository
    ```bash
    git clone git@github.com:Shashank110205/Adobe_Round1B.git
    ```

2. Move into the repository folder
    ```bash
    cd Adobe_Round1B
    ```

3. Build the docker image(requires internet connection)
    ```bash
    docker compose build
    ```

    or build the image manually

    ```bash
    docker build -t pdf-processor .
    ```

4. Run the docker image(works offline)
    ```bash
    docker run -it --rm \
      --name pdf-processor \
      -v "<path_to_the_input_directory>:/app/input" \
      pdf-processor
    ```

>[!NOTE]
> The structure of the input directory should be as follows:
>```
> .
> ├── challenge1b_input.json
> ├── challenge1b_output.json
> └── PDFs
>     ├── pdf1.pdf
>     ├── pdf2.pdf
>     ├── pdf3.pdf
>     ├── ...
>     ├── pdf16.pdf
> ```
