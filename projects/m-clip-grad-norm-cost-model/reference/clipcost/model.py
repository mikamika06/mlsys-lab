class ParameterGroup:
    def __init__(self, name: str, numel: int, dtype_bytes: int):
        self.name = name
        self.numel = numel
        self.dtype_bytes = dtype_bytes

    def compute_cost(self) -> float:
        return float(self.numel * self.dtype_bytes)

    def to_dict(self):
        return {
            "name": self.name,
            "numel": self.numel,
            "dtype_bytes": self.dtype_bytes
        }


class ClipCostModel:
    def __init__(self, groups):
        self.groups = [
            g if isinstance(g, ParameterGroup) else ParameterGroup(**g)
            for g in groups
        ]

    def total_bytes(self) -> int:
        return sum(g.compute_cost() for g in self.groups)

    def estimated_time_us(self, bandwidth_gbps: float) -> float:
        b = self.total_bytes()
        bytes_per_sec = bandwidth_gbps * 1e9
        if bytes_per_sec == 0:
            return 0.0
        return (b / bytes_per_sec) * 1e6
