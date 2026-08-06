import math

def quantize_fixed_point(arr: list[float], frac_bits: int) -> list[int]:
    """Quantizes a float array to fixed‑point with round‑half‑to‑even."""
    scale = 1 << frac_bits

    def process_val(val_f: float) -> int:
        val = val_f * scale
        f = math.floor(val)
        frac = val - f
        if frac > 0.5:
            res = f + 1
        elif frac < 0.5:
            res = f
        else:
            if f % 2 == 0:
                res = f
            else:
                res = f + 1
        return int(res)

    def recurse(item):
        if isinstance(item, list):
            return [recurse(sub) for sub in item]
        else:
            return process_val(float(item))

    return recurse(arr)
