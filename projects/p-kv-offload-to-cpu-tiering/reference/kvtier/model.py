class CostModel:
    def __init__(self, bandwidth_gbps: float):
        self.bw = bandwidth_gbps

    def transfer_cost(self, num_bytes: int) -> float:
        if self.bw <= 0:
            return float("inf")
        return num_bytes / (self.bw * 1e9)
