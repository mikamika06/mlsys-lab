class CostModel:
    def __init__(self, bandwidth_gbps: float):
        raise NotImplementedError

    def transfer_cost(self, num_bytes: int) -> float:
        raise NotImplementedError
