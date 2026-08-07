# Long-Context Serving: KV Cache Memory Floor and Concurrency Ceiling

## Problem Description
Our long-context LLM serving cluster is experiencing unpredictable Out-Of-Memory (OOM) failures and request rejections when scaling context lengths from 32k up to 128k and 256k tokens. Operational monitoring reveals two root causes:

1. **Inaccurate Max Context Bounds**: Serving instances fail to properly parse extended RoPE scaling configurations (`rope_scaling` in HuggingFace configs), either underestimating available sequence length or failing to apply model overrides (`override_max_model_len` or `max_model_len`).
2. **KV Cache & Concurrency Miscalculations**: The memory planner underestimates the static KV cache allocation floor required per request at extreme context lengths (128k/256k) and miscalculates single-GPU KV cache allocation under Tensor Parallelism (TP).

We need a structured capacity module `kvcapacity` to calculate context limits, per-request KV cache floors, GPU concurrency ceilings, and serving feasibility across hardware and numerical precision combinations.

## Required Package Structure
You must implement the following functions across four files:

### `kvcapacity/rope.py`
- `compute_effective_context(config: dict) -> int`
  - Determines the maximum supported context length for a given model config.
  - Base length is `config.get("max_position_embeddings", 2048)`.
  - If `rope_scaling` is a dictionary in `config`:
    - If `"factor"` is present and not `None`, scaling factor is `float(rope_scaling["factor"])`, making effective context `int(base_len * factor)`.
    - Else if `"original_max_position_embeddings"` is present and not `None`, if `base_len <= orig`, effective context is `orig`.
  - If `"override_max_model_len"` is in `config` and not `None`, effective context is `min(effective, int(config["override_max_model_len"]))`.
  - Otherwise, if `"max_model_len"` is in `config` and not `None`, effective context is `min(effective, int(config["max_model_len"]))`.
  - Returns `max(1, effective_context)`.

### `kvcapacity/floor.py`
- `get_dtype_bytes(dtype: str) -> float`
  - Returns element byte size: `"float32"` -> 4.0, `"float16"` / `"bfloat16"` -> 2.0, `"fp8"` / `"fp8_e4m3fn"` / `"fp8_e5m2"` / `"int8"` -> 1.0, `"int4"` -> 0.5.
  - Raises `ValueError` for unknown data types.
- `per_request_kv_bytes(model_config: dict, seq_len: int, kv_dtype: str = "float16") -> int`
  - Calculates total KV cache memory required across all layers for ONE request with length `seq_len`.
  - Formula: $2 \times \text{num\_layers} \times \text{num\_kv\_heads} \times \text{head\_dim} \times \text{seq\_len} \times \text{bytes\_per\_elem}$.
  - Keys: `num_hidden_layers` (or `num_layers`), `num_key_value_heads` (or `num_attention_heads`), `head_dim` (or `hidden_size // num_attention_heads`).
- `model_weights_bytes(model_config: dict, model_dtype: str = "float16") -> int`
  - Computes model parameter memory in bytes. If `"num_parameters"` is present, uses `num_parameters * bytes_per_elem`. Otherwise, estimates parameters from transformer layer dimensions (embedding + per-layer attn & mlp + output head).

### `kvcapacity/feasibility.py`
- `concurrency_ceiling(gpu_memory_gb: float, model_config: dict, tp_size: int, model_dtype: str, kv_dtype: str, seq_len: int, gpu_memory_utilization: float = 0.9) -> int`
  - Computes maximum concurrent requests of length `seq_len` that a single GPU in a TP cluster of size `tp_size` can hold in its KV cache pool.
  - If `num_key_value_heads % tp_size != 0` or `tp_size <= 0`, return `0`.
  - Total usable GPU memory = `gpu_memory_gb * (1024**3) * gpu_memory_utilization`.
  - Per-GPU weight memory = `model_weights_bytes(model_config, model_dtype) / tp_size`.
  - Per-GPU KV memory pool = `usable_memory - weight_memory_per_gpu`.
  - Per-GPU single-request KV memory = `per_request_kv_bytes(model_config, seq_len, kv_dtype) / tp_size`.
  - Concurrency ceiling = `int(per_gpu_kv_pool // per_gpu_single_request_kv)`. Returns `max(0, ceiling)`.
- `build_feasibility_matrix(...) -> list[dict]`
  - Evaluates combinations of GPUs, TP options, model dtypes, and KV dtypes for a given target sequence length, returning dicts with fields `gpu_name`, `gpu_memory_gb`, `tp_size`, `model_dtype`, `kv_dtype`, `seq_len`, `concurrency_ceiling`, and `feasible` (`concurrency_ceiling >= 1`).

### `tests/test_regression.py`
- Write unit test assertions verifying that tensor parallelism properly reduces per-GPU KV cache footprint and that invalid TP head splits yield concurrency 0.
