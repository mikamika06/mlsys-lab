LOGS = [
    "cmake -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AMX=ON ..",
    "cmake -DGGML_AVX512=ON -DGGML_AMX=OFF ..",
    "cmake -DGGML_AVX512=OFF -DGGML_AMX=OFF ..",
]

CONFIGS = [
    {"log": LOGS[0], "expected_flags": {"GGML_AVX512": True, "GGML_AVX512_VBMI": True, "GGML_AMX": True}},
    {"log": LOGS[1], "expected_flags": {"GGML_AVX512": True, "GGML_AVX512_VBMI": False, "GGML_AMX": False}},
    {"log": LOGS[2], "expected_flags": {"GGML_AVX512": False, "GGML_AVX512_VBMI": False, "GGML_AMX": False}},
]

def parse_isa_flags(log_str):
    res = {}
    for token in log_str.split():
        if token.startswith("-DGGML_"):
            parts = token[2:].split("=")
            if len(parts) == 2:
                key, val = parts
                res[key] = (val.upper() == "ON")
    return res

def compare_performance(amx_tps, avx_tps):
    speedup = amx_tps / max(avx_tps, 1e-6)
    return {"speedup": float(speedup), "efficient": bool(speedup > 1.1)}

def classify_format(fmt_name):
    f = fmt_name.upper()
    if "AMX" in f or "Q8_0" in f or "Q5_K" in f:
        return "accelerated"
    if "Q4_0" in f:
        return "standard"
    return "fallback"
