import random

CONFIGS = [
    {
        "step1": {"name": "tokenizer", "inputs": ["TEXT"], "outputs": ["INPUT_IDS", "ATTENTION_MASK"]},
        "step2": {"name": "transformer", "inputs": ["INPUT_IDS", "ATTENTION_MASK"], "outputs": ["LOGITS"]}
    },
    {
        "step1": {"name": "preprocessor", "inputs": ["RAW_IMAGE"], "outputs": ["PIXELS"]},
        "step2": {"name": "detector", "inputs": ["PIXELS"], "outputs": ["BOXES", "SCORES"]}
    },
    {
        "step1": {"name": "embedder", "inputs": ["TOKENS"], "outputs": ["EMBEDDINGS"]},
        "step2": {"name": "classifier", "inputs": ["EMBEDDINGS"], "outputs": ["CLASSES"]}
    }
]

LATENCY_DATA = [
    {
        "step1_latencies": [10.0, 12.0, 11.0, 15.0, 14.0],
        "step2_latencies": [20.0, 22.0, 21.0, 25.0, 24.0]
    },
    {
        "step1_latencies": [5.0, 6.0, 5.5, 7.0, 8.0],
        "step2_latencies": [40.0, 42.0, 41.0, 45.0, 50.0]
    }
]

ERROR_STRINGS = [
    "Error: step 'transformer' input 'INPUT_IDS' is not produced by any preceding step or model input.",
    "Error: tensor data type mismatch between step1 output 'PIXELS' (FP32) and step2 input 'PIXELS' (FP16).",
    "Error: cyclic dependency detected in ensemble pipeline configuration."
]

def build_wiring(cfg):
    return {
        "inputs": cfg["step1"]["inputs"],
        "outputs": cfg["step2"]["outputs"],
        "steps": [
            {"name": cfg["step1"]["name"], "input_map": {i: i for i in cfg["step1"]["inputs"]}, "output_map": {o: o for o in cfg["step1"]["outputs"]}},
            {"name": cfg["step2"]["name"], "input_map": {i: i for i in cfg["step2"]["inputs"]}, "output_map": {o: o for o in cfg["step2"]["outputs"]}}
        ]
    }

def compute_latencies(data):
    s1 = sorted(data["step1_latencies"])
    s2 = sorted(data["step2_latencies"])
    combined = sorted([a + b for a, b in zip(s1, s2)])
    n = len(combined)
    p50 = combined[n // 2]
    p99 = combined[int(0.99 * (n - 1))]
    return {"p50": float(p50), "p99": float(p99)}

def classify_error(err_str):
    if "input" in err_str and "not produced" in err_str:
        return "MISSING_TENSOR_INPUT"
    if "data type mismatch" in err_str:
        return "TYPE_MISMATCH"
    if "cyclic dependency" in err_str:
        return "CYCLIC_DEPENDENCY"
    return "UNKNOWN"
