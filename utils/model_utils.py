import numpy as np
from onnxruntime import InferenceSession
from tokenizers import Tokenizer
from numpy import dot
from numpy.linalg import norm

# Load MiniLM tokenizer and model
minilm_tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
minilm_session = InferenceSession("model/model.onnx")

# Load BGE tokenizer and model
bge_tokenizer = Tokenizer.from_file("tokenizer/bge_tokenizer/tokenizer.json")
bge_session = InferenceSession("model/bge_model/model.onnx")

# --- Utility functions ---

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return dot(a, b) / (norm(a) * norm(b) + 1e-10)

def cosine_similarity_batch(query_vec, matrix):
    return (matrix @ query_vec) / (norm(matrix, axis=1) * norm(query_vec) + 1e-10)

def batch_encode(tokenizer, texts, max_length=128):
    input_ids = []
    attention_masks = []

    for text in texts:
        tokens = tokenizer.encode(text)
        ids = tokens.ids[:max_length]
        mask = [1] * len(ids)

        pad_len = max_length - len(ids)
        ids += [0] * pad_len
        mask += [0] * pad_len

        input_ids.append(ids)
        attention_masks.append(mask)

    return np.array(input_ids, dtype=np.int64), np.array(attention_masks, dtype=np.int64)

# --- MiniLM ---

def get_embedding_minilm(text: str) -> np.ndarray:
    input_ids, attention_mask = batch_encode(minilm_tokenizer, [text])
    outputs = minilm_session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })
    return np.mean(outputs[0][0], axis=0)

def get_minilm_embeddings_batch(texts: list[str], max_length=128) -> np.ndarray:
    input_ids, attention_mask = batch_encode(minilm_tokenizer, texts, max_length)
    outputs = minilm_session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })
    return np.mean(outputs[0], axis=1)

# --- BGE ---

def get_embedding_bge(text: str) -> np.ndarray:
    input_ids, attention_mask = batch_encode(bge_tokenizer, [text], max_length=512)
    outputs = bge_session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })
    return np.mean(outputs[0][0], axis=0)

def get_bge_embeddings_batch(texts: list[str], max_length=512) -> np.ndarray:
    input_ids, attention_mask = batch_encode(bge_tokenizer, texts, max_length)
    outputs = bge_session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })
    return np.mean(outputs[0], axis=1)

# --- Combined Similarity (Single) ---

def get_combined_similarity(text1: str, text2: str, w_minilm: float = 0.5, w_bge: float = 0.5) -> float:
    emb1_minilm = get_embedding_minilm(text1)
    emb2_minilm = get_embedding_minilm(text2)
    sim_minilm = cosine_similarity(emb1_minilm, emb2_minilm)

    emb1_bge = get_embedding_bge(text1)
    emb2_bge = get_embedding_bge(text2)
    sim_bge = cosine_similarity(emb1_bge, emb2_bge)

    return w_minilm * sim_minilm + w_bge * sim_bge

# --- Combined Similarity (Batch) ---

def get_combined_similarity_batch(query: str, texts: list[str], w_minilm: float = 0.5, w_bge: float = 0.5) -> np.ndarray:
    q_minilm = get_minilm_embeddings_batch([query])[0]
    q_bge = get_bge_embeddings_batch([query])[0]

    emb_minilm = get_minilm_embeddings_batch(texts)
    emb_bge = get_bge_embeddings_batch(texts)

    sim_minilm = cosine_similarity_batch(q_minilm, emb_minilm)
    sim_bge = cosine_similarity_batch(q_bge, emb_bge)

    return w_minilm * sim_minilm + w_bge * sim_bge
