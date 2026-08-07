import numpy as np

OOM_TEST_CASES = [
    {
        "model_size_bytes": 4 * 1024 * 1024 * 1024,
        "kv_bytes_per_token": 128 * 1024,
        "memory_limit_bytes": 8 * 1024 * 1024 * 1024,
        "context_lengths": [2048, 8192, 16384, 32768, 65536]
    },
    {
        "model_size_bytes": 7 * 1024 * 1024 * 1024,
        "kv_bytes_per_token": 256 * 1024,
        "memory_limit_bytes": 12 * 1024 * 1024 * 1024,
        "context_lengths": [4096, 8192, 16384, 32768]
    }
]

def simulate_oom(case):
    model_sz = case["model_size_bytes"]
    kv_rate = case["kv_bytes_per_token"]
    limit = case["memory_limit_bytes"]
    results = []
    for ctx in case["context_lengths"]:
        total = model_sz + ctx * kv_rate
        oom = total > limit
        results.append({"context": ctx, "total_bytes": total, "oom": oom})
    return results

MODELFILE_CASES = [
    {
        "text": "FROM llama3:8b-instruct-q4_K_M\nSYSTEM \"You are a helpful assistant.\"\nPARAMETER temperature 0.7\n",
        "expected_system": "You are a helpful assistant.",
        "expected_quant": "q4_k_m"
    },
    {
        "text": "FROM mistral:7b-v0.3-q8_0\nSYSTEM 'Be concise.'\n",
        "expected_system": "Be concise.",
        "expected_quant": "q8_0"
    }
]

def parse_modelfile_ref(text):
    system = None
    quant = None
    from_base = None
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("FROM"):
            from_base = line.split(maxsplit=1)[1].strip()
            if "-" in from_base:
                parts = from_base.split("-")
                quant = parts[-1].lower()
        elif line.startswith("SYSTEM"):
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                system = parts[1].strip().strip('"').strip("'")
    return {"from": from_base, "system": system, "quant": quant}

def verify_modelfile_ref(parsed, expected_system, expected_quant):
    s_match = parsed.get("system") == expected_system
    q_match = parsed.get("quant") == expected_quant.lower()
    return bool(s_match and q_match)

SERVER_CASES = [
    {
        "lm_studio": {"text": "Hello world", "tokens": 10, "latency_ms": 120.5},
        "mlx": {"text": "Hello world!", "tokens": 10, "latency_ms": 110.0}
    },
    {
        "lm_studio": {"text": "Compute unit test", "tokens": 25, "latency_ms": 300.0},
        "mlx": {"text": "Compute unit test", "tokens": 25, "latency_ms": 280.0}
    }
]

def compare_servers_ref(lm_res, mlx_res):
    text_match = lm_res["text"].strip().rstrip("!") == mlx_res["text"].strip().rstrip("!")
    token_match = lm_res["tokens"] == mlx_res["tokens"]
    latency_ratio = mlx_res["latency_ms"] / lm_res["latency_ms"]
    return {
        "text_match": text_match,
        "token_match": token_match,
        "latency_ratio": float(latency_ratio)
    }
