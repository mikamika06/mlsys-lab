import math

FP16_MAX = 65504.0


def fp16_overflow_tail_probability(std: float) -> float:
    if std <= 0.0:
        return 0.0
    z = FP16_MAX / std
    p_exceed_positive = 0.5 * math.erfc(z / math.sqrt(2.0))
    return 2.0 * p_exceed_positive
