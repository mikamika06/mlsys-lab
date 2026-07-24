class RecomputeCounter:
    def __init__(self, data):
        self.data = list(data)

    @property
    def sum(self):
        return sum(self.data)
