from transformers import AutoTokenizer
from optimum.exporters.onnx import main_export

model_id = "google/flan-t5-small"

# ✅ Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained("model/flan-t5/tokenizer")

# ✅ Export ONNX model
main_export(
    model_name_or_path=model_id,
    output="model/flan-t5",
    task="text2text-generation",
    opset=17,
    use_past=True  # ✅ THIS IS IMPORTANT
)
