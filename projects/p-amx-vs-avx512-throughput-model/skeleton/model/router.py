class ExecutionRouter:
    def __init__(self, avx_model, amx_model):
        raise NotImplementedError

    def select_isa(self, M: int, N: int, K: int, dtype: str) -> str:
        raise NotImplementedError

    def find_k_crossover(self, M: int, N: int, dtype: str, k_max: int = 2048) -> int:
        raise NotImplementedError
