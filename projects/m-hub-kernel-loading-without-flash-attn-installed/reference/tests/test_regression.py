import sys
sys.path.insert(0, ".")
from hf_attn.router import resolve_backend, dispatch_attention, AttentionInterface


def test_fallback_when_flash_missing():
    config = {
        "attn_implementation": "flash",
        "fallback_priority": ["flash", "sdpa", "math"]
    }
    flash_backend = AttentionInterface("flash", available=False)
    sdpa_backend = AttentionInterface("sdpa", available=True)
    math_backend = AttentionInterface("math", available=True)
    backends = [flash_backend, sdpa_backend, math_backend]

    resolved = resolve_backend(config, backends)
    assert resolved == "sdpa", f"Expected sdpa fallback, got {resolved}"

    out = dispatch_attention(config, None, None, None, backends)
    assert out == "executed_sdpa", f"Expected executed_sdpa, got {out}"
