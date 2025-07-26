from transformers import AutoTokenizer, AutoModel
from transformers.onnx import export
from pathlib import Path
from transformers.onnx.features import FeaturesManager
import torch

# Model name
model_name = "intfloat/e5-small-v2"

# Load model and tokenizer
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("model/e5-small-v2-onnx/tokenizer")

# Get the model class
model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model)
onnx_config = model_onnx_config(model.config)

output_path = Path("model/e5-small-v2-onnx/model.onnx")
output_path.parent.mkdir(parents=True, exist_ok=True)

export(
    preprocessor=tokenizer,
    model=model,
    config=onnx_config,
    opset=17,
    output=output_path
)


print("✅ ONNX export completed to model/e5-small-v2-onnx/")