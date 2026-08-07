from model.throughput import ThroughputModel


class ExecutionRouter:
    def __init__(self, avx_model: ThroughputModel, amx_model: ThroughputModel):
        self.avx_model = avx_model
        self.amx_model = amx_model

    def select_isa(self, M: int, N: int, K: int, dtype: str) -> str:
        c_avx = self.avx_model.predict_cycles(M, N, K, dtype)
        c_amx = self.amx_model.predict_cycles(M, N, K, dtype)
        if c_amx < c_avx:
            return "amx"
        return "avx512"

    def find_k_crossover(self, M: int, N: int, dtype: str, k_max: int = 2048) -> int:
        for k in range(1, k_max + 1):
            if self.select_isa(M, N, k, dtype) == "amx":
                return k
        return -1
