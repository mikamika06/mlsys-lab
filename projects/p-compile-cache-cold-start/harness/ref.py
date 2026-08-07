def get_oracle_trace(inputs):
    return {"trace_ok": 1.0, "compiled_count": len(inputs)}

def get_oracle_cache(hits, total):
    return {"cache_hit_ratio": float(hits) / float(total) if total > 0 else 0.0}

def get_oracle_transfer(payload):
    return {"transfer_ok": 1.0 if isinstance(payload, bytes) and len(payload) > 0 else 0.0}

def get_oracle_warmup(warmed):
    return {"warmup_ok": 1.0 if warmed else 0.0}

def get_oracle_request_cost(cost):
    return {"first_request_cost": float(cost)}
