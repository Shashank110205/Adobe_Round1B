from transformers import AutoTokenizer, AutoModel
from optimum.exporters.onnx import main_export
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
onnx_dir = Path(".")

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(onnx_dir)
tokenizer.backend_tokenizer.save("tokenizer.json")

# Export ONNX
main_export(
    model_name_or_path=MODEL_NAME,
    output=onnx_dir,
    task="feature-extraction",
    framework="pt",
    opset=17,
)
