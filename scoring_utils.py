# scoring_utils.py

import numpy as np
import re

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / norm_product if norm_product != 0 else 0.0

def extract_keywords(text: str) -> list:
    # Simple dynamic keyword extraction: keep nouns and long words
    words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())  # words ≥ 5 letters
    return list(set(words))[:10]  # return top 10 unique keywords

def keyword_match_score(text: str, keywords: list) -> float:
    return sum(1 for k in keywords if k in text.lower()) / len(keywords) if keywords else 0.0

def compute_final_score(sim: float, kw_score: float) -> float:
    return 0.8 * sim + 0.2 * kw_score
