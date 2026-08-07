class PreemptionPolicy:
    def __init__(self, compute_bw, pcie_bw, bytes_per_tok, compute_ovh, pcie_ovh):
        self.compute_bw = float(compute_bw)
        self.pcie_bw = float(pcie_bw)
        self.bytes_per_tok = float(bytes_per_tok)
        self.compute_ovh = float(compute_ovh)
        self.pcie_ovh = float(pcie_ovh)

    def recompute_time(self, seq_len: int) -> float:
        raise NotImplementedError

    def swap_time(self, seq_len: int) -> float:
        raise NotImplementedError

    def breakeven_seq_len(self) -> int:
        raise NotImplementedError

    def decide(self, seq_len: int) -> str:
        raise NotImplementedError

    def evaluate_trace(self, trace: list[int], mode: str = "smart") -> list[float]:
        raise NotImplementedError
