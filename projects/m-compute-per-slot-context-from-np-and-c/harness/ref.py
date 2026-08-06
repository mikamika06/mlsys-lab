import numpy as np

def generate_context_cases():
    return [
        {"c_total": 8192, "np_slots": 4, "req": 1024},
        {"c_total": 16384, "np_slots": 8, "req": 2048},
        {"c_total": 4096, "np_slots": 16, "req": 512},
        {"c_total": 32768, "np_slots": 2, "req": 8192},
    ]

def generate_saturation_cases():
    return [
        {"model_bytes_base": 10 * 1024**3, "slot_overhead_bytes": 1024, "max_memory_bytes": 16 * 1024**3, "c_total": 8192},
        {"model_bytes_base": 20 * 1024**3, "slot_overhead_bytes": 2048, "max_memory_bytes": 32 * 1024**3, "c_total": 16384},
        {"model_bytes_base": 8 * 1024**3, "slot_overhead_bytes": 512, "max_memory_bytes": 10 * 1024**3, "c_total": 4096},
    ]

def generate_metrics_cases():
    return [
        "llamacpp:prompt_tokens_processed_total 500\nllamacpp:prompt_tokens_cached_total 500",
        "llamacpp:prompt_tokens_processed_total 1000\nllamacpp:prompt_tokens_cached_total 0",
        "llamacpp:prompt_tokens_processed_total 0\nllamacpp:prompt_tokens_cached_total 1000",
    ]
