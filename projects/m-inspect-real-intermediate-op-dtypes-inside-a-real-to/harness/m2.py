import sys

def check(workdir):
    out = {"returns_tensors": 0.0, "fp16_overflows": 0.0, "bf16_safe": 0.0}
    sys.path.insert(0, workdir)
    try:
        from autocast_inspect.inspector import synthesize_overflow
    except ImportError:
        out["_note"] = "Could not import synthesize_overflow"
        return out

    try:
        res = synthesize_overflow()
        if not isinstance(res, tuple) or len(res) != 2:
            out["_note"] = "synthesize_overflow must return a tuple of 2 tensors"
            return out
        a, b = res
    except Exception as e:
        out["_note"] = f"Error running synthesize_overflow: {e}"
        return out

    out["returns_tensors"] = 1.0

    fp16_val = (a.half() * b.half()).sum().item()
    if fp16_val == float('inf') or fp16_val == float('-inf'):
        out["fp16_overflows"] = 1.0

    bf16_val = (a.bfloat16() * b.bfloat16()).sum().item()
    if bf16_val != float('inf') and bf16_val != float('-inf'):
        out["bf16_safe"] = 1.0

    return out
