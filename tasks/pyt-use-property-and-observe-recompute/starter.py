import functools

class RecomputeCounter:
    def __init__(self, data):
        self.data = list(data)

    @functools.cached_property
    def sum(self):
        return sum(self.data)
