import random

class RefPolicy:
    def __init__(self, compute_bw, pcie_bw, bytes_per_tok, compute_ovh, pcie_ovh):
        self.compute_bw = float(compute_bw)
        self.pcie_bw = float(pcie_bw)
        self.bytes_per_tok = float(bytes_per_tok)
        self.compute_ovh = float(compute_ovh)
        self.pcie_ovh = float(pcie_ovh)

    def recompute_time(self, seq_len: int) -> float:
        return self.compute_ovh + (seq_len / self.compute_bw)

    def swap_time(self, seq_len: int) -> float:
        return 2.0 * (self.pcie_ovh + (seq_len * self.bytes_per_tok) / self.pcie_bw)

    def breakeven_seq_len(self) -> int:
        a = self.compute_ovh
        b = 1.0 / self.compute_bw
        c = 2.0 * self.pcie_ovh
        d = 2.0 * self.bytes_per_tok / self.pcie_bw

        if b == d:
            return 0 if c >= a else 999999999

        return max(0, int((c - a) / (b - d)))

    def decide(self, seq_len: int) -> str:
        if self.recompute_time(seq_len) <= self.swap_time(seq_len):
            return "recompute"
        return "swap"


def generate_trace(n, seed=42):
    random.seed(seed)
    return [random.randint(10, 1000) for _ in range(n)]

def p99(latencies):
    if not latencies:
        return 0.0
    s = sorted(latencies)
    idx = int(0.99 * len(s))
    if idx >= len(s):
        idx = len(s) - 1
    return s[idx]
