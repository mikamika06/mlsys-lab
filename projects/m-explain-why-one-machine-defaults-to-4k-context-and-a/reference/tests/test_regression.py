import sys
sys.path.insert(0, ".")
from runner.memory import predict_resident_bytes


def test_cliff_detection():
    model_bytes = 4 * 1024**3
    kv_per_token = 512 * 1024
    ctx_len = 4096
    parallel = 4
    vram_bytes = 6 * 1024**3
    resident = predict_resident_bytes(model_bytes, kv_per_token, ctx_len, parallel, vram_bytes)
    assert resident > vram_bytes, "failed to detect memory cliff when KV cache exceeds available VRAM"
