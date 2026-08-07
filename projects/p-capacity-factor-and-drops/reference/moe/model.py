class MoEModelSimulator:
    def __init__(self, num_experts: int, capacity_factor: float):
        self.num_experts = num_experts
        self.cf = capacity_factor

    def forward(self, tokens, drop_tokens=True):
        import numpy as np
        tokens = np.asarray(tokens)
        n_tokens = len(tokens)
        cap = int(max(1, self.cf * n_tokens / self.num_experts))
        choices = np.random.randint(0, self.num_experts, size=n_tokens)
        counts = np.bincount(choices, minlength=self.num_experts)
        dropped = 0
        computed = 0
        for c in counts:
            if c > cap:
                if drop_tokens:
                    dropped += (c - cap)
                    computed += cap
                else:
                    computed += c
            else:
                computed += c
        return {"total": n_tokens, "computed": computed, "dropped": dropped, "capacity": cap}
