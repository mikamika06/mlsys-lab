def kahan_sum_fp16(a):
    total = 0.0
    compensation = 0.0
    for value in a:
        x = float(value)
        y = x - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    return float(total)
