# adobe_round1b/scoring_utils.py

import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / norm_product if norm_product != 0 else 0.0

def extract_keywords(text: str, top_n: int = 10) -> list:
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])
    sorted_keywords = sorted(scores, key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_keywords[:top_n]]

def keyword_match_score(text: str, keywords: list) -> float:
    if not keywords:
        return 0.0
    matched = sum(1 for k in keywords if k.lower() in text.lower())
    return matched / len(keywords)

def compute_final_score(sim: float, kw_score: float) -> float:
    return 0.8 * sim + 0.2 * kw_score
