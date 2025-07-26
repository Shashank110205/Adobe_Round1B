import re
import numpy as np
from collections import defaultdict

# Remove the problematic import
# We'll handle embeddings through dependency injection instead

def refine_text(text: str, max_length: int = 300) -> str:
    """Extract meaningful content adaptive to context"""
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Find natural sentence endings
    sentences = []
    start = 0
    in_quote = False
    
    for i, char in enumerate(text):
        if char in {'"', "'"}:
            in_quote = not in_quote
            
        # Sentence-ending punctuation not in quotes
        if char in '.!?' and not in_quote and (i == len(text)-1 or text[i+1] in ' \t\n'):
            sentence = text[start:i+1].strip()
            if 3 < len(sentence.split()) < 40:  # Reasonable sentence length
                sentences.append(sentence)
                start = i+1
                
        # Stop if we have enough content
        if len(" ".join(sentences)) > max_length:
            break
    
    # Return full sentences
    result = " ".join(sentences).strip()
    
    # Fallback if no good sentences found
    if not result:
        if len(text) > max_length:
            return text[:max_length].rsplit('.', 1)[0] + '.'  # End at last sentence
        return text
    
    return result

def extract_keywords(text: str, top_n: int = 15) -> list:
    """Extract contextually important keywords"""
    words = re.findall(r'\b\w{3,}\b', text.lower())
    if not words:
        return []
    
    freq = defaultdict(int)
    for word in words:
        freq[word] += 1
    
    # Get top keywords by frequency
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:top_n]]

def keyword_match_score(text: str, keywords: list) -> float:
    """Contextual keyword matching"""
    if not keywords:
        return 0.0
        
    text_words = set(re.findall(r'\b\w+\b', text.lower()))
    matches = text_words & set(keywords)
    return min(len(matches) / len(keywords), 1.0)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity safely and return a float"""
    # Ensure vectors are 1-dimensional
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    norm_product = norm1 * norm2
    
    if norm_product < 1e-10:  # Avoid division by zero
        return 0.0
        
    return float(dot_product / norm_product)

# Modified to accept embedder instance
def generate_contextual_title(text: str, query_keywords: list, original_title: str, embedder) -> str:
    """Generate context-aware title using semantic analysis"""
    # Use original title if it's meaningful
    if original_title and len(original_title.split()) > 2 and not re.search(r'page \d+', original_title, re.I):
        return original_title
    
    # Split text into candidate phrases
    phrases = re.split(r'[.!?]', text)
    if not phrases:
        return "Relevant Content"
    
    # Get embeddings for query keywords
    keyword_embeddings = []
    for keyword in query_keywords:
        keyword_embeddings.append(embedder.get_embedding(keyword, "query:"))
    
    avg_keyword_embedding = np.mean(keyword_embeddings, axis=0) if keyword_embeddings else None
    
    # Find best matching phrase
    best_phrase = ""
    best_score = -1
    
    for phrase in phrases:
        if len(phrase.split()) < 3 or len(phrase.split()) > 12:
            continue
            
        # Calculate relevance score
        phrase_embedding = embedder.get_embedding(phrase, "query:")
        if avg_keyword_embedding is not None:
            semantic_score = cosine_similarity(avg_keyword_embedding, phrase_embedding)
        else:
            semantic_score = 0
            
        # Count keyword matches
        phrase_words = set(re.findall(r'\w+', phrase.lower()))
        keyword_score = len(phrase_words & set(query_keywords)) / max(len(query_keywords), 1)
        
        # Combined score
        total_score = (semantic_score * 0.7) + (keyword_score * 0.3)
        
        if total_score > best_score:
            best_score = total_score
            best_phrase = phrase.strip()
    
    # Fallback strategies
    if best_phrase:
        # Shorten to title length
        words = best_phrase.split()[:8]
        return ' '.join(words)
    elif phrases:
        # Use first meaningful phrase
        return phrases[0].strip()[:80]
    else:
        return "Key Information"