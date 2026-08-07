import hashlib


class OptimizationProfile:
    def __init__(self, name="default"):
        self.name = name
        self.shapes = {}

    def add_shape(self, name, min_shape, opt_shape, max_shape):
        if len(min_shape) != len(opt_shape) or len(opt_shape) != len(max_shape):
            raise ValueError("Dimensions must match")

        for mi, op, ma in zip(min_shape, opt_shape, max_shape):
            if not (mi <= op <= ma):
                raise ValueError("Inconsistent min <= opt <= max bounds")

        self.shapes[name] = {
            "min": tuple(min_shape),
            "opt": tuple(opt_shape),
            "max": tuple(max_shape),
        }

    def validate_shape(self, name, shape):
        if name not in self.shapes:
            return False
        bounds = self.shapes[name]
        if len(shape) != len(bounds["min"]):
            return False
        for s, mi, ma in zip(shape, bounds["min"], bounds["max"]):
            if s < mi or s > ma:
                return False
        return True

    def profile_hash(self):
        sorted_keys = sorted(self.shapes.keys())
        parts = []
        for k in sorted_keys:
            v = self.shapes[k]
            parts.append(f"{k}:{v['min']}:{v['opt']}:{v['max']}")
        return hashlib.sha256(";".join(parts).encode("utf-8")).hexdigest()
