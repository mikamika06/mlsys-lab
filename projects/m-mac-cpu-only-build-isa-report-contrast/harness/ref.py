REPORTS = [
    {
        "id": 0,
        "log": "CMAKE_ARGS='-DGGML_NATIVE=ON'\nDetected Neon: True\nDetected FP16: True\nDetected BF16: True\n",
        "manual": {"neon": True, "fp16": True, "bf16": True, "dotprod": True},
        "native": {"neon": True, "fp16": True, "bf16": False, "dotprod": False}
    },
    {
        "id": 1,
        "log": "CMAKE_ARGS='-DGGML_NATIVE=OFF -DGGML_NEON=ON'\nDetected Neon: True\nDetected FP16: False\nDetected BF16: False\n",
        "manual": {"neon": True, "fp16": False, "bf16": False, "dotprod": True},
        "native": {"neon": False, "fp16": False, "bf16": False, "dotprod": False}
    },
    {
        "id": 2,
        "log": "CMAKE_ARGS='-DGGML_NATIVE=ON -DGGML_FP16=ON'\nDetected Neon: True\nDetected FP16: True\nDetected BF16: True\n",
        "manual": {"neon": True, "fp16": True, "bf16": True, "dotprod": True},
        "native": {"neon": True, "fp16": True, "bf16": True, "dotprod": True}
    }
]

def parse_log(log_str):
    features = {}
    for line in log_str.strip().split("\n"):
        if "Detected" in line:
            parts = line.split(":")
            if len(parts) == 2:
                k = parts[0].replace("Detected", "").strip().lower()
                v = parts[1].strip() == "True"
                features[k] = v
        elif "CMAKE_ARGS" in line:
            if "GGML_NATIVE=ON" in line:
                features["native_flag"] = True
            else:
                features["native_flag"] = False
    return features

def contrast_reports(native_rep, manual_rep):
    diffs = {}
    all_keys = set(native_rep.keys()).union(set(manual_rep.keys()))
    for k in all_keys:
        nv = native_rep.get(k, False)
        mv = manual_rep.get(k, False)
        if nv != mv:
            diffs[k] = {"native": nv, "manual": mv}
    return diffs

def analyze_tier(features):
    if features.get("fp16") and features.get("neon"):
        return "T1"
    return "T0"
