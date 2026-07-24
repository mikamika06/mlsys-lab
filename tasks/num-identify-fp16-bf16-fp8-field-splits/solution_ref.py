def identify_fp_formats():
    """
    Return a dictionary mapping floating‑point format names to tuples of
    (exp_bits, mantissa_bits, bias). The bias is computed as 2^(e-1)-1.
    """
    specs = {
        'fp16':  (5, 10),
        'bf16':  (8, 7),
        'E4M3':  (4, 3),
        'E5M2':  (5, 2)
    }
    result = {}
    for name, (exp_bits, mantissa_bits) in specs.items():
        bias = 2 ** (exp_bits - 1) - 1
        result[name] = (exp_bits, mantissa_bits, bias)
    return result
