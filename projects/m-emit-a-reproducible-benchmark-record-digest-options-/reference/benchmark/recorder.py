import hashlib
import json


def emit_record(options, phases):
    phase_toks = {}
    for p in phases:
        toks_per_sec = p["tokens"] / p["elapsed"] if p["elapsed"] > 0 else 0.0
        phase_toks[p["name"]] = {
            "tokens": p["tokens"],
            "elapsed": p["elapsed"],
            "tok_per_sec": toks_per_sec,
        }
    payload = {"options": options, "phases": phase_toks}
    canonical = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {"digest": digest, "options": options, "phases": phase_toks}
