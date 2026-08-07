class PreemptionPolicy:
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

        val = (c - a) / (b - d)
        return max(0, int(val))

    def decide(self, seq_len: int) -> str:
        if self.recompute_time(seq_len) <= self.swap_time(seq_len):
            return "recompute"
        return "swap"

    def evaluate_trace(self, trace: list[int], mode: str = "smart") -> list[float]:
        res = []
        for s in trace:
            if mode == "smart":
                action = self.decide(s)
            else:
                action = mode

            if action == "recompute":
                res.append(self.recompute_time(s))
            else:
                res.append(self.swap_time(s))
        return res
