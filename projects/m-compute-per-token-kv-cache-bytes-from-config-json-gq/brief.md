# KV Cache Sizing, Context Solver, and Block Prediction

A downstream engine setup team is running into out-of-memory errors and inaccurate startup memory allocations when configuring LLM deployments under vLLM. They report three primary issues:

1. They cannot reliably calculate the per-token KV cache memory footprint across different arch configurations (such as Llama, Qwen, and Mistral variants using Grouped-Query Attention) given only raw `config.json` dictionaries.
2. Production capacity planning requires calculating the maximum supported context length (`max_model_len`) given a strict per-GPU VRAM overhead ceiling, model parameters, and target KV cache space, but current estimations drift by multiple megabytes.
3. Node provisioning scripts fail to accurately predict the total number of GPU KV cache blocks (`num_gpu_blocks`) that vLLM will report upon startup for custom configurations of model dtype, total memory, GPU memory utilization fraction, and block size.

Your task is to build a low-level memory planning library and a set of regression tests that address these issues.

## Requirements

### Package Structure
Write your implementation inside `vllm_budget/`:
- `vllm_budget/kv.py`: `bytes_per_token(config, dtype)`
- `vllm_budget/solver.py`: `max_context_length(config, dtype, model_weight_bytes, non_model_overhead_bytes, total_vram_bytes)`
- `vllm_budget/blocks.py`: `predict_num_gpu_blocks(config, dtype, total_vram_bytes, gpu_memory_utilization, model_weight_bytes, non_model_overhead_bytes, block_size)`
- `tests/test_regression.py`: A regression test suite validating these invariants against subtle boundary errors.

### Calculations

1. **`bytes_per_token(config, dtype)`**:
   - `config` is a dictionary containing LLM config parameters (`num_hidden_layers`, `num_key_value_heads`, `hidden_size`, `num_attention_heads`, `head_dim`). If `head_dim` is not explicitly provided, it defaults to `hidden_size // num_attention_heads`. If `num_key_value_heads` is not explicitly provided, it defaults to `num_attention_heads` (MHA).
   - Each token stores both Key and Value tensors across all layers.
   - Per key/value element size in bytes is determined by `dtype` (`float32` / `fp32` = 4, `float16` / `fp16` / `bfloat16` / `bf16` = 2, `int8` / `fp8` = 1).
   - Calculation: `2 * num_hidden_layers * num_key_value_heads * head_dim * bytes_per_element`.

2. **`max_context_length(config, dtype, model_weight_bytes, non_model_overhead_bytes, total_vram_bytes)`**:
   - Computes the maximum context length (integer tokens) that fits in available KV cache VRAM budget.
   - Available budget for KV cache: `available = total_vram_bytes - model_weight_bytes - non_model_overhead_bytes`.
   - If `available <= 0`, return `0`. Otherwise, return `floor(available / bytes_per_token)`.

3. **`predict_num_gpu_blocks(config, dtype, total_vram_bytes, gpu_memory_utilization, model_weight_bytes, non_model_overhead_bytes, block_size)`**:
   - Total VRAM managed by vLLM is `usable_vram = floor(total_vram_bytes * gpu_memory_utilization)`.
   - Available KV cache memory space is `kv_budget = usable_vram - model_weight_bytes - non_model_overhead_bytes`.
   - If `kv_budget <= 0`, return `0`.
   - Single block byte cost is `block_bytes = bytes_per_token * block_size`.
   - Returns `floor(kv_budget / block_bytes)`.
