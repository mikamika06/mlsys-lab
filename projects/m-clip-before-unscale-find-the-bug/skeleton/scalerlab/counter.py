class UnderflowTracker:
    def __init__(self):
        raise NotImplementedError

    def update(self, found_inf):
        raise NotImplementedError

    def get_skipped_count(self):
        raise NotImplementedError
