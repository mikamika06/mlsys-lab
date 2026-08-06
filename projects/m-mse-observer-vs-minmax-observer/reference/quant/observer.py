import numpy as np


class MinMaxObserver:
    def __init__(self, bits: int = 8, symmetric: bool = True):
        self.bits = bits
        self.symmetric = symmetric
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def update(self, x: np.ndarray) -> None:
        self.min_val = min(self.min_val, float(np.min(x)))
        self.max_val = max(self.max_val, float(np.max(x)))

    def compute_params(self) -> tuple[float, int]:
        qmin = -(1 << (self.bits - 1)) if self.symmetric else 0
        qmax = (1 << (self.bits - 1)) - 1 if self.symmetric else (1 << self.bits) - 1

        if self.symmetric:
            max_abs = max(abs(self.min_val), abs(self.max_val))
            scale = max_abs / max(1, qmax) if max_abs > 0 else 1.0
            zero_point = 0
        else:
            span = self.max_val - self.min_val
            scale = span / max(1, qmax - qmin) if span > 0 else 1.0
            zero_point = int(round(-self.min_val / scale) + qmin)
            zero_point = max(qmin, min(qmax, zero_point))

        return float(scale), int(zero_point)


class MSEObserver:
    def __init__(self, bits: int = 8, symmetric: bool = True, num_bins: int = 100):
        self.bits = bits
        self.symmetric = symmetric
        self.num_bins = num_bins
        self.data = []

    def update(self, x: np.ndarray) -> None:
        self.data.append(x.copy())

    def compute_params(self) -> tuple[float, int]:
        if not self.data:
            return 1.0, 0
        cat = np.concatenate([d.flatten() for d in self.data])
        qmin = -(1 << (self.bits - 1)) if self.symmetric else 0
        qmax = (1 << (self.bits - 1)) - 1 if self.symmetric else (1 << self.bits) - 1

        min_v, max_v = float(np.min(cat)), float(np.max(cat))
        if self.symmetric:
            limit = max(abs(min_v), abs(max_v))
            best_limit = limit
            best_mse = float("inf")
            for p in np.linspace(0.5, 1.0, self.num_bins):
                lim = limit * p
                scale = lim / max(1, qmax)
                if scale == 0:
                    continue
                q = np.clip(np.round(cat / scale), qmin, qmax)
                dequant = q * scale
                mse = float(np.mean((cat - dequant) ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best_limit = lim
            scale = best_limit / max(1, qmax) if best_limit > 0 else 1.0
            return float(scale), 0
        else:
            span = max_v - min_v
            best_scale = span / max(1, qmax - qmin) if span > 0 else 1.0
            best_zp = qmin
            best_mse = float("inf")
            for p in np.linspace(0.5, 1.0, self.num_bins):
                s = (span * p) / max(1, qmax - qmin)
                if s == 0:
                    continue
                for zp in range(qmin, qmax + 1):
                    dequant = (np.clip(np.round(cat / s) + zp, qmin, qmax) - zp) * s
                    mse = float(np.mean((cat - dequant) ** 2))
                    if mse < best_mse:
                        best_mse = mse
                        best_scale = s
                        best_zp = zp
            return float(best_scale), int(best_zp)
