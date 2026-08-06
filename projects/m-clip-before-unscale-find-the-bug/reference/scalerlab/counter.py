class UnderflowTracker:
    """Tracks skipped optimizer steps due to gradient underflow/overflow."""
    def __init__(self):
        self.skipped_steps = 0
        self.total_steps = 0

    def update(self, found_inf):
        self.total_steps += 1
        if found_inf:
            self.skipped_steps += 1

    def get_skipped_count(self):
        return self.skipped_steps
