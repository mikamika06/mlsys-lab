class DropTracker:
    def __init__(self, capacity_factor: float):
        raise NotImplementedError

    def update(self, routings, expert_capacities):
        raise NotImplementedError

    def drop_ratio(self):
        raise NotImplementedError
