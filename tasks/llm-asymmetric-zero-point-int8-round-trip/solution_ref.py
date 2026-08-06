def asymmetric_quant_round_trip(x: list[float]) -> tuple[list[float], int]:
    """Asymmetric quantization round trip."""
    min_val = float(x[0])
    max_val = float(x[0])
    for v in x:
        v_f = float(v)
        if v_f < min_val:
            min_val = v_f
        if v_f > max_val:
            max_val = v_f

    if max_val == min_val:
        s = 1.0
    else:
        s = (max_val - min_val) / 255.0

    z_float = -min_val / s
    zp = int(round(z_float))
    if zp < -128:
        zp = -128
    elif zp > 127:
        zp = 127

    deq = []
    for i in range(len(x)):
        q_val = int(round(float(x[i]) / s + zp))
        if q_val < -128:
            q_val = -128
        elif q_val > 127:
            q_val = 127
        deq.append((float(q_val) - zp) * s)

    return deq, int(zp)
