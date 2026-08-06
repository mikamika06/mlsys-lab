import struct

LOGS = [
    ("error: failed to load model: unknown architecture 'custom-llama'", "missing_architecture"),
    ("ggml_cuda_init: failed to initialize CUDA: out of memory", "cuda_out_of_memory"),
    ("error: model file incompatible, expected version 3, got 1", "unsupported_gguf_version"),
    ("terminate called after throwing an instance of 'std::bad_alloc'", "host_out_of_memory"),
    ("error: tensor 'blk.0.attn_q.weight' has wrong shape", "tensor_shape_mismatch"),
    ("error: invalid tokenizer vocab size 32000 does not match tensor dimensions", "vocab_size_mismatch"),
    ("Segmentation fault (core dumped) at ggml_compute_forward_attn", "segmentation_fault"),
    ("error: rope freq base not found in metadata", "missing_rope_metadata"),
    ("error: unsupported quantization type Q4_K_M_EX", "unsupported_quantization"),
    ("error: failed to map file: mmap failed with error 12", "mmap_failure"),
    ("error: context window 16384 exceeds model max length 4096", "context_overflow"),
    ("error: invalid magic number for GGUF file", "invalid_magic"),
    ("error: sliding window pattern mismatch in layer 12", "sliding_window_error"),
    ("error: fp16 compute not supported on this device", "compute_capability_error"),
    ("error: missing EOS token in tokenizer config", "missing_eos_token"),
    ("error: rope scaling type 'linear' unknown", "unknown_rope_scaling"),
    ("error: tensor data offset out of bounds", "tensor_offset_out_of_bounds"),
    ("error: custom pooling layer not implemented in ggama", "unimplemented_layer"),
    ("error: kv cache allocation failed for 32 layers", "kv_cache_allocation_failure"),
    ("error: shared expert count mismatch in MoE block", "moe_config_mismatch")
]

FIXES = {
    "missing_architecture": "python3 -m gguf_triage.repair --fix-arch",
    "cuda_out_of_memory": "export GGML_CUDA_FORCE_MMQ=1",
    "unsupported_gguf_version": "llama-convert-legacy-to-gguf",
    "host_out_of_memory": "ulimit -v unlimited",
    "tensor_shape_mismatch": "git submodule update --init --recursive",
    "vocab_size_mismatch": "python3 -m gguf_triage.repair --fix-vocab",
    "segmentation_fault": "export GGML_SCHED_MAX_COPIES=1",
    "missing_rope_metadata": "python3 -m gguf_triage.repair --fix-rope",
    "unsupported_quantization": "llama-quantize --allow-requantize",
    "mmap_failure": "export LLAMA_NO_MMAP=1",
    "context_overflow": "llama-cli -c 4096",
    "invalid_magic": "file model.gguf",
    "sliding_window_error": "python3 -m gguf_triage.repair --fix-window",
    "compute_capability_error": "export GGML_CUDA_NO_PEER=1",
    "missing_eos_token": "python3 -m gguf_triage.repair --fix-eos",
    "unknown_rope_scaling": "python3 -m gguf_triage.repair --fix-scaling",
    "tensor_offset_out_of_bounds": "llama-repair-tensor-offsets",
    "unimplemented_layer": "export GGML_BACKEND_GUARD=0",
    "kv_cache_allocation_failure": "llama-cli -b 512",
    "moe_config_mismatch": "python3 -m gguf_triage.repair --fix-moe"
}

def make_sample_gguf():
    buf = bytearray()
    buf.extend(b"GGUF")
    buf.extend(struct.pack("<I", 3))
    buf.extend(struct.pack("<Q", 0))
    buf.extend(struct.pack("<Q", 0))
    return bytes(buf)
