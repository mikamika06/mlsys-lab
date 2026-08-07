def evaluate_policy(requests, policy_fn, cost_model, gamma):
    tpts = []
    for r in requests:
        b = r["b"]
        p = r["p_true"]
        td, tt, tv = cost_model(b, gamma)

        use_spec = policy_fn(r["domain"], b)
        if use_spec:
            if p >= 1.0:
                e_toks = 1.0 + gamma
            else:
                e_toks = 1.0 + (p - p**(gamma + 1)) / (1.0 - p)
            tpts.append((gamma * td + tv) / e_toks)
        else:
            tpts.append(tt)

    if not tpts:
        return 0.0
    tpts.sort()
    idx = int(0.95 * len(tpts))
    return tpts[idx]

class AdaptivePolicy:
    def __init__(self, cost_model, gamma, default_p=0.5):
        self.cost_model = cost_model
        self.gamma = gamma
        self.default_p = default_p
        self.stats = {}

    def update(self, domain, drafted, accepted):
        if domain not in self.stats:
            self.stats[domain] = {"d": 0, "a": 0}
        self.stats[domain]["d"] += drafted
        self.stats[domain]["a"] += accepted

    def decide(self, domain, b):
        if domain in self.stats and self.stats[domain]["d"] > 0:
            p_est = self.stats[domain]["a"] / self.stats[domain]["d"]
        else:
            p_est = self.default_p

        td, tt, tv = self.cost_model(b, self.gamma)

        if p_est >= 1.0:
            e_toks = 1.0 + self.gamma
        else:
            e_toks = 1.0 + (p_est - p_est**(self.gamma + 1)) / (1.0 - p_est)

        t_spec = self.gamma * td + tv
        return t_spec < e_toks * tt
