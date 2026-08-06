def validate_hlo_compilation(hlo_text):
    if "mismatch" in hlo_text or "invalid" in hlo_text:
        raise RuntimeError("XLA compile error: shape mismatch")
    return True
