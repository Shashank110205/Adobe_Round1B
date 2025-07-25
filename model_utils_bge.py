# model_utils.py

import numpy as np
from onnxruntime import InferenceSession
from tokenizers import Tokenizer
from numpy import dot
from numpy.linalg import norm

# Load BGE tokenizer and model
bge_tokenizer = Tokenizer.from_file("tokenizer/bge_tokenizer/tokenizer.json")
bge_session = InferenceSession("model/bge_model/model.onnx")

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return dot(a, b) / (norm(a) * norm(b) + 1e-10)

def get_embedding_bge(text: str, instruction: str = None) -> np.ndarray:
    if instruction:
        text = f"{instruction}: {text}"

    tokens = bge_tokenizer.encode(text)

    max_len = 512
    input_ids = tokens.ids[:max_len]
    input_ids += [0] * (max_len - len(input_ids))

    attention_mask = [1 if i < len(tokens.ids[:max_len]) else 0 for i in range(max_len)]

    input_ids = np.array([input_ids], dtype=np.int64)
    attention_mask = np.array([attention_mask], dtype=np.int64)

    outputs = bge_session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })

    return np.mean(outputs[0][0], axis=0)

def get_combined_similarity(text1: str, text2: str) -> float:
    emb1 = get_embedding_bge(text1)
    emb2 = get_embedding_bge(text2)
    return cosine_similarity(emb1, emb2)

# Optional: batching (if needed later)
def batch_encode(tokenizer, texts, max_length=512):
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
