import numpy as np


def convert_peft_to_gguf(peft_dict: dict, alpha: float) -> dict:
    gguf_dict = {
        "metadata": {
            "adapter.lora.alpha": float(alpha)
        },
        "tensors": {}
    }
    for key, val in peft_dict.items():
        arr = np.asarray(val, dtype=np.float32)
        clean_key = key.replace("base_model.model.", "").replace(".default", "")
        if ".lora_A.weight" in clean_key:
            gguf_key = clean_key.replace(".lora_A.weight", ".lora_a")
        elif ".lora_B.weight" in clean_key:
            gguf_key = clean_key.replace(".lora_B.weight", ".lora_b")
        else:
            gguf_key = clean_key
        gguf_dict["tensors"][gguf_key] = arr
    return gguf_dict
