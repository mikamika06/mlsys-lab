def classify_fp_value(x: float) -> dict:
    """
    Classify how the real value x would be stored in fp16 and in bf16.

    Return {"fp16": <cls>, "bf16": <cls>} where each <cls> is one of:
      - "overflow": abs(x) exceeds the format's largest finite magnitude
        (the value would round to +/-inf).
      - "underflow": x is nonzero but abs(x) is smaller than the format's
        smallest positive subnormal (the value would flush to +/-0).
      - "ok": otherwise (representable finite and nonzero, or exactly 0).
    """
    raise NotImplementedError('your code here')
