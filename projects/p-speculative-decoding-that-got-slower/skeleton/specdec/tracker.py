class AcceptanceTracker:
    def __init__(self, window_size=100):
        raise NotImplementedError

    def record(self, domain, n_accepted, gamma):
        raise NotImplementedError

    def get_acceptance_rate(self, domain=None):
        raise NotImplementedError

    def get_stats(self):
        raise NotImplementedError
