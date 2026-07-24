def solve_quadratic(a: float, b: float, c: float) -> tuple[float, float]:
    disc = b * b - 4.0 * a * c
    sqrt_disc = disc ** 0.5
    if b >= 0:
        q = -(b + sqrt_disc) / 2.0
    else:
        q = -(b - sqrt_disc) / 2.0
    return q / a, c / q
