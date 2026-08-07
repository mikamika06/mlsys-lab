from __future__ import annotations


def optimize_zero_point(W: list[float], scale: float, bits: int, iters: int) -> tuple[list[float], int]:
    qmax = (1 << bits) - 1
    z = 0

    def reconstruct(z_value: int) -> list[float]:
        return [
            scale * (max(0, min(qmax, round(w / scale) + z_value)) - z_value)
            for w in W
        ]

    for _ in range(iters):
        best_z = z
        best_err = None
        for candidate in range(z - 2, z + 3):
            rec = reconstruct(candidate)
            err = sum((w - r) ** 2 for w, r in zip(W, rec))
            if best_err is None or err < best_err:
                best_err = err
                best_z = candidate
        z = best_z

    return reconstruct(z), z
