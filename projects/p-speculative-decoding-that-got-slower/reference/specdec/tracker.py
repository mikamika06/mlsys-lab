from collections import deque


class AcceptanceTracker:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.global_history = deque(maxlen=window_size)
        self.domain_history = {}

    def record(self, domain, n_accepted, gamma):
        if gamma <= 0:
            return
        rate = float(n_accepted) / float(gamma)
        self.global_history.append(rate)
        if domain not in self.domain_history:
            self.domain_history[domain] = deque(maxlen=self.window_size)
        self.domain_history[domain].append(rate)

    def get_acceptance_rate(self, domain=None):
        if domain is None:
            if not self.global_history:
                return 0.0
            return sum(self.global_history) / len(self.global_history)
        if domain not in self.domain_history or not self.domain_history[domain]:
            return self.get_acceptance_rate(domain=None)
        return sum(self.domain_history[domain]) / len(self.domain_history[domain])

    def get_stats(self):
        domains = list(self.domain_history.keys())
        stats = {
            "global_rate": self.get_acceptance_rate(None),
            "domains": {d: self.get_acceptance_rate(d) for d in domains}
        }
        return stats
