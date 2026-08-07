from model.isa import ISAParameters

class ThroughputModel:
    def __init__(self, isa: ISAParameters):
        raise NotImplementedError

    def compute_tile_dimensions(self, M: int, N: int, K: int, dtype: str) -> dict:
        raise NotImplementedError

    def predict_cycles(self, M: int, N: int, K: int, dtype: str) -> float:
        raise NotImplementedError

    def predict_gflops(self, M: int, N: int, K: int, dtype: str, clock_ghz: float = 2.0) -> float:
        raise NotImplementedError
