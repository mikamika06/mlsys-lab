import re

LOGS = [
    """INFO 08-12 10:00:00 llm_engine.py:73] Initializing an LLM engine (v0.6.0) with config: model='meta-llama/Llama-2-7b-hf', tensor_parallel_size=2, dtype=torch.float16, quantization=none
    INFO 08-12 10:00:02 weight_utils.py:150] Loading weights finished in 2.14 seconds.
    INFO 08-12 10:00:03 paged_attention_ops.py:100] Using FlashAttention backend.""",
    """INFO 08-12 11:00:00 llm_engine.py:73] Initializing an LLM engine (v0.6.3) with config: model='Qwen/Qwen2-7B-Instruct', tensor_parallel_size=4, dtype=torch.bfloat16, quantization=awq
    INFO 08-12 11:00:05 weight_utils.py:150] Loading weights finished in 4.50 seconds."""
]

def parse_log(log_text):
    model = None
    tp = None
    quant = None
    version = None
    m_ver = re.search(r"vLLM engine \(v([0-9\.]+)\)", log_text)
    if m_ver:
        version = m_ver.group(1)
    m_cfg = re.search(r"model='([^']+)', tensor_parallel_size=(\d+), .*?quantization=(\w+)", log_text)
    if m_cfg:
        model = m_cfg.group(1)
        tp = int(m_cfg.group(2))
        quant = m_cfg.group(3)
    return {"model": model, "tensor_parallel_size": tp, "quantization": quant, "version": version}

SHARDING_TESTS = [
    {"num_attention_heads": 32, "num_kv_heads": 8, "tensor_parallel_size": 2, "expected": True},
    {"num_attention_heads": 32, "num_kv_heads": 6, "tensor_parallel_size": 4, "expected": False},
    {"num_attention_heads": 16, "num_kv_heads": 2, "tensor_parallel_size": 8, "expected": True},
    {"num_attention_heads": 14, "num_kv_heads": 2, "tensor_parallel_size": 4, "expected": False}
]

def check_sharding(num_attention_heads, num_kv_heads, tensor_parallel_size):
    if num_attention_heads % tensor_parallel_size != 0:
        return False
    if num_kv_heads % tensor_parallel_size != 0 and tensor_parallel_size % num_kv_heads != 0:
        return False
    return True

DIAG_TESTS = [
    {"tp": 1, "sharding_valid": True, "quant": "awq", "symptom": "garbage", "diagnosis": "none"},
    {"tp": 2, "sharding_valid": False, "quant": "awq", "symptom": "garbage", "diagnosis": "invalid_sharding"},
    {"tp": 2, "sharding_valid": True, "quant": "none", "symptom": "garbage", "diagnosis": "column_parallel_weight_mismatch"}
]

def diagnose_garbage(tp, sharding_valid, quant, symptom):
    if symptom != "garbage":
        return "ok"
    if tp == 1:
        return "single_gpu_ok"
    if not sharding_valid:
        return "invalid_sharding"
    return "column_parallel_weight_mismatch"
