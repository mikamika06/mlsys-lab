def kahan_sum_fp16(a):
    # TODO: this accumulates directly in float16 and loses low-order bits.
    total = 0
    for value in a:
        total = total + value
    return float(total)
===== END
