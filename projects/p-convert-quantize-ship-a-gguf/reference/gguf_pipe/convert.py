import json
import os


def verify_tokenizer(vocab_path):
    if not os.path.exists(vocab_path):
        return False
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return "vocab" in data and len(data["vocab"]) > 0


def convert_safetensors_to_gguf(weights_dir, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    meta = {"magic": "GGUF", "version": 3, "source": weights_dir}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return output_path
