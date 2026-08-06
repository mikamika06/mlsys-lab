class ParameterGroup:
    def __init__(self, name: str, numel: int, dtype_bytes: int):
        raise NotImplementedError

    def compute_cost(self) -> float:
        raise NotImplementedError


class ClipCostModel:
    def __init__(self, groups: list):
        raise NotImplementedError

    def total_bytes(self) -> int:
        raise NotImplementedError

    def estimated_time_us(self, bandwidth_gbps: float) -> float:
        raise NotImplementedError
