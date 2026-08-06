def predict_resident_bytes(model_bytes, kv_bytes_per_token, ctx_len, parallel, vram_bytes):
    total_kv = kv_bytes_per_token * ctx_len * parallel
    resident = model_bytes + total_kv
    return resident
