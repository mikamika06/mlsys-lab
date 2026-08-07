import os
import sys

sys.path.insert(0, os.path.abspath("reference"))

from kvcapacity.rope import compute_effective_context
from kvcapacity.floor import get_dtype_bytes, per_request_kv_bytes, model_weights_bytes
from kvcapacity.feasibility import concurrency_ceiling, build_feasibility_matrix

CONFIGS = [
    {
        "name": "llama-2-7b",
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "hidden_size": 4096,
        "max_position_embeddings": 4096,
        "num_parameters": 6738415616,
    },
    {
        "name": "llama-3-8b-instruct",
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "max_position_embeddings": 8192,
        "rope_scaling": {
            "type": "llama3",
            "factor": 16.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
        "num_parameters": 8030261248,
    },
    {
        "name": "llama-3-70b-instruct",
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "hidden_size": 8192,
        "max_position_embeddings": 8192,
        "rope_scaling": {"type": "llama3", "factor": 16.0},
        "override_max_model_len": 131072,
        "num_parameters": 70589132800,
    },
    {
        "name": "custom-long-context-128k",
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "hidden_size": 3584,
        "max_position_embeddings": 32768,
        "rope_scaling": {"type": "linear", "factor": 4.0},
        "num_parameters": 7000000000,
    },
]

GPU_CONFIGS = [
    {"name": "A100-40GB", "memory_gb": 40.0},
    {"name": "A100-80GB", "memory_gb": 80.0},
    {"name": "H100-80GB", "memory_gb": 80.0},
    {"name": "L40S-48GB", "memory_gb": 48.0},
]

TP_OPTIONS = [1, 2, 4, 8]
MODEL_DTYPES = ["float16", "bfloat16", "fp8"]
KV_DTYPES = ["float16", "fp8", "int8"]
