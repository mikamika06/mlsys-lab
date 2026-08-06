import hashlib
import json

RUNS = [
    {
        "options": {"batch_size": 4, "max_tokens": 128, "warmup": 2},
        "phases": [
            {"name": "prefill", "tokens": 512, "elapsed": 0.05},
            {"name": "decode", "tokens": 512, "elapsed": 0.25},
        ],
    },
    {
        "options": {"batch_size": 8, "max_tokens": 256, "warmup": 4},
        "phases": [
            {"name": "prefill", "tokens": 1024, "elapsed": 0.08},
            {"name": "decode", "tokens": 2048, "elapsed": 0.90},
        ],
    },
]


def generate_record(run_config):
    opts = run_config["options"]
    phases = run_config["phases"]
    phase_toks = {}
    for p in phases:
        toks_per_sec = p["tokens"] / p["elapsed"] if p["elapsed"] > 0 else 0.0
        phase_toks[p["name"]] = {
            "tokens": p["tokens"],
            "elapsed": p["elapsed"],
            "tok_per_sec": toks_per_sec,
        }
    payload = {"options": opts, "phases": phase_toks}
    canonical = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {"digest": digest, "options": opts, "phases": phase_toks}


def analyze_flaws(script_text):
    flaws = []
    if "warmup" not in script_text or "time.sleep" in script_text:
        flaws.append({"flaw": "cold_start_included", "bias": "positive"})
    if "print(" in script_text or "sys.stdout.write" in script_text:
        flaws.append({"flaw": "synchronous_logging", "bias": "negative"})
    if "queue" in script_text:
        flaws.append({"flaw": "unbounded_queue", "bias": "positive"})
    return sorted(flaws, key=lambda x: x["flaw"])


def quantify_cold_start(cold_elapsed, steady_elapsed, total_tokens):
    cold_toks = total_tokens / cold_elapsed
    steady_toks = total_tokens / steady_elapsed
    inflation = ((cold_toks - steady_toks) / steady_toks) * 100.0
    return inflation
