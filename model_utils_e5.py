import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from numpy import dot
from numpy.linalg import norm
from functools import lru_cache

# Initialize ONNX session with optimizations
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = 1  # Helps with CPU contention

# Load tokenizer and model with caching
@lru_cache(maxsize=1)
def get_tokenizer():
    return Tokenizer.from_file("model/e5-small-v2-onnx/tokenizer/tokenizer.json")

@lru_cache(maxsize=1)
def get_session():
    return ort.InferenceSession(
        "model/e5-small-v2-onnx/model.onnx",
        sess_options=session_options,
        providers=['CPUExecutionProvider']
    )

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Optimized cosine similarity with epsilon for division stability"""
    return dot(a, b) / max(norm(a) * norm(b), 1e-10)

def get_embedding_e5(text: str, instruction: str = "query:") -> np.ndarray:
    """
    Get embedding with instruction prefix, optimized for batch processing
    Args:
        text: Input text to embed
        instruction: "query:" or "passage:" as per E5 instructions
    Returns:
        Normalized embedding vector
    """
    # Prepare input
    full_text = f"{instruction} {text.strip()}"
    tokenizer = get_tokenizer()
    session = get_session()
    
    # Tokenize with fixed length
    encoding = tokenizer.encode(full_text)
    seq_len = min(len(encoding.ids), 512)
    
    # Prepare inputs
    input_ids = np.zeros((1, 512), dtype=np.int64)
    attention_mask = np.zeros((1, 512), dtype=np.int64)
    token_type_ids = np.zeros((1, 512), dtype=np.int64)
    
    # Fill the available sequence
    input_ids[0, :seq_len] = encoding.ids[:seq_len]
    attention_mask[0, :seq_len] = 1
    
    # Run inference
    outputs = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
    )
    
    # Mean pooling with attention mask
    last_hidden_state = outputs[0]  # shape: (1, seq_len, hidden_size)
    input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
    sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
    sum_mask = np.maximum(np.sum(input_mask_expanded, axis=1), 1e-9)
    embedding = sum_embeddings / sum_mask
    
    # Normalize
    embedding = embedding / np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-10)
    return embedding[0]  # Return first (and only) embedding