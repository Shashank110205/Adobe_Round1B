import numpy as np
import onnxruntime as ort

# from transformers import AutoTokenizer

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained("./model/flan-t5/tokenizer")

# # Load ONNX sessions
# encoder_sess = ort.InferenceSession("./model/flan-t5/encoder_model.onnx")
# decoder_sess = ort.InferenceSession("./model/flan-t5/decoder_model.onnx")

# def generate_title(prompt: str, max_length=20):
#     inputs = tokenizer(prompt, return_tensors="np")

#     # Ensure inputs are in the correct format
#     inputs["input_ids"] = inputs["input_ids"].astype(np.int64)
#     inputs["attention_mask"] = inputs["attention_mask"].astype(np.int64)
#     # Run encoder
#     encoder_outputs = encoder_sess.run(
#         output_names=None,
#         input_feed={
#             "input_ids": inputs["input_ids"],
#             "attention_mask": inputs["attention_mask"]
#         }
#     )

#     encoder_hidden_states = encoder_outputs[0]

#     # Start decoding
#     decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)

#     for _ in range(max_length):
#         decoder_outputs = decoder_sess.run(
#             output_names=["logits"],
#             input_feed={
#                 "input_ids": decoder_input_ids,
#                 "encoder_hidden_states": encoder_hidden_states,
#                 "encoder_attention_mask": inputs["attention_mask"]
#             }
#         )
#         logits = decoder_outputs[0]
#         next_token_id = int(np.argmax(logits[:, -1, :], axis=-1))
#         decoder_input_ids = np.append(decoder_input_ids, [[next_token_id]], axis=1)

#         if next_token_id == tokenizer.eos_token_id:
#             break

#     decoded = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
#     return decoded

def generate_title(text: str) -> str:
    lines = text.split('\n')
    for line in lines:
        if len(line.strip()) > 10 and line.strip()[0].isupper():
            return line.strip()
    return text[:30] + "..."