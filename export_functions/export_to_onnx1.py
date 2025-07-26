from transformers import AutoTokenizer, AutoModel
from optimum.onnxruntime import ORTModelForFeatureExtraction
from optimum.exporters.onnx import main_export

model_id = "BAAI/bge-small-en-v1.5"

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained("tokenizer/bge_tokenizer")

# Export model
main_export(
    model_name_or_path=model_id,
    task="feature-extraction",
    output="model/bge_model",
    opset=17
)
