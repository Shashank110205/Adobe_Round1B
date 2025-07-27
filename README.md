# 🧠 Adobe Hackathon Round 1B — Persona-Aware PDF Section Ranker

A solution to extract and rank the most relevant PDF sections for a given **persona** and **job-to-be-done**, using optimized ONNX-based embeddings and contextual keyword matching.

---

## 📌 Problem Statement

> Given multiple PDFs, a **persona** (e.g., travel agent, researcher) and their **task**, extract the top 5 most relevant sections across documents — and return them in a structured JSON format.

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

