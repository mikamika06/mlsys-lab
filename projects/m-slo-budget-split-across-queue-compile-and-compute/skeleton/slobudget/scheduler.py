class SLOBudgetScheduler:
    """Schedules requests by dropping those that cannot complete within the SLO budget."""

    def __init__(self, planner):
        raise NotImplementedError

    def filter_batch(self, batch, current_time_ms):
        raise NotImplementedError

    def simulate_pipeline(self, requests, compute_cost_fn):
        raise NotImplementedError
