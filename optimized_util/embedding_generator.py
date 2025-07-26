import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from functools import lru_cache
# Add this import at the top
import numpy as np


class EmbeddingGenerator:
    def __init__(self):
        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.session_options.intra_op_num_threads = 1
        
        # Initialize tokenizer and session
        self.tokenizer = self._get_tokenizer()
        self.session = self._get_session()
        self.cache = {}
    
    @staticmethod
    @lru_cache(maxsize=1)
    def _get_tokenizer():
        return Tokenizer.from_file("model/e5-small-v2-onnx/tokenizer/tokenizer.json")
    
    @staticmethod
    @lru_cache(maxsize=1)
    def _get_session():
        return ort.InferenceSession(
            "model/e5-small-v2-onnx/model.onnx",
            sess_options=ort.SessionOptions(),
            providers=['CPUExecutionProvider']
        )
    
    def get_embedding(self, text: str, instruction: str = "query:") -> np.ndarray:
        """Get embedding for any text with instruction prefix"""
        # Prepare input
        full_text = f"{instruction} {text.strip()}"
        
        # Check cache
        cache_key = f"{instruction}:{text[:100]}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Tokenize
        encoding = self.tokenizer.encode(full_text)
        seq_len = min(len(encoding.ids), 512)
        
        # Prepare inputs
        input_ids = np.zeros((1, 512), dtype=np.int64)
        attention_mask = np.zeros((1, 512), dtype=np.int64)
        token_type_ids = np.zeros((1, 512), dtype=np.int64)
        
        input_ids[0, :seq_len] = encoding.ids[:seq_len]
        attention_mask[0, :seq_len] = 1
        
        # Run inference
        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
        )
        
        # Mean pooling
        last_hidden_state = outputs[0]
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
        sum_mask = np.maximum(np.sum(input_mask_expanded, axis=1), 1e-9)
        embedding = sum_embeddings / sum_mask
        
        # Normalize
        embedding = embedding / np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-10)
        result = embedding[0]
        
        # Cache result
        self.cache[cache_key] = result
        return result
    

    # Then modify the get_embeddings_batch method:
    def get_embeddings_batch(self, texts: list, instruction: str) -> list:
        """Batch process embeddings efficiently"""
        # Separate cached and uncached texts
        cached_embeddings = []
        uncached_texts = []
        cache_keys = []
        
        for text in texts:
            cache_key = f"{instruction}:{text[:100]}"
            if cache_key in self.cache:
                cached_embeddings.append(self.cache[cache_key])
            else:
                uncached_texts.append(text)
                cache_keys.append(cache_key)
        
        if uncached_texts:
            # Prepare batch inputs
            batch_size = len(uncached_texts)
            input_ids = np.zeros((batch_size, 512), dtype=np.int64)
            attention_mask = np.zeros((batch_size, 512), dtype=np.int64)
            token_type_ids = np.zeros((batch_size, 512), dtype=np.int64)
            
            # Tokenize batch
            for i, text in enumerate(uncached_texts):
                full_text = f"{instruction} {text.strip()}"
                encoding = self.tokenizer.encode(full_text)
                seq_len = min(len(encoding.ids), 512)
                input_ids[i, :seq_len] = encoding.ids[:seq_len]
                attention_mask[i, :seq_len] = 1
            
            # Run batch inference
            outputs = self.session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids
                }
            )
            
            # Process batch results
            last_hidden_state = outputs[0]
            input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
            sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
            sum_mask = np.maximum(np.sum(input_mask_expanded, axis=1), 1e-9)
            batch_embeddings = sum_embeddings / sum_mask
            
            # Normalize
            norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            batch_embeddings /= np.maximum(norms, 1e-10)
            
            # Cache results and ensure 1D vectors
            for i, embedding in enumerate(batch_embeddings):
                # Ensure embedding is 1-dimensional
                flat_embedding = embedding.flatten()
                self.cache[cache_keys[i]] = flat_embedding
                cached_embeddings.append(flat_embedding)
        
        return cached_embeddings