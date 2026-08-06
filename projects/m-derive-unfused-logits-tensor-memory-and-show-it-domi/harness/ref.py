from logits.memory import unfused_logits_bytes, weight_memory_bytes, logits_dominates_weights
from logits.chunked import chunked_crossentropy_bytes, memory_savings_ratio
from logits.kernel import fused_logits_forward_ref, fused_logits_forward_triton

CONFIGS = [
    {"batch_size": 2, "seq_len": 4096, "vocab_size": 32000, "num_params": 7e9, "chunk_size": 1024},
    {"batch_size": 4, "seq_len": 2048, "vocab_size": 128000, "num_params": 8e9, "chunk_size": 2048},
    {"batch_size": 1, "seq_len": 8192, "vocab_size": 32000, "num_params": 1.5e9, "chunk_size": 512},
]
