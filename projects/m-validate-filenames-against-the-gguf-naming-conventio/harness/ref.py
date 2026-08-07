VALID_FILENAMES = [
    "llama-3-8b-instruct-q4_k_m.gguf",
    "mistral-7b-f16.gguf",
    "phi-3-mini-q8_0.gguf"
]

INVALID_FILENAMES = [
    "llama-3-8b.gguf",
    "model-q4_k_m.bin",
    "invalid_name_f32"
]

SAMPLE_GGUF = {
    "metadata": {"general.architecture": "llama", "general.file_type": 2},
    "tensors": {
        "token_embd.weight": [[0.1, 0.2], [0.3, 0.4]]
    }
}

HUB_META = {"general.architecture": "llama", "block_count": 32}
LOCAL_META_GOOD = {"general.architecture": "llama", "block_count": 32}
LOCAL_META_BAD = {"general.architecture": "llama", "block_count": 24}
