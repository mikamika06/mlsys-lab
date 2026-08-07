class SpikeDetector:
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.history = []

    def update(self, loss):
        self.history.append(loss)
        if len(self.history) < 5:
            return False
        recent = self.history[-5:-1]
        mean = sum(recent) / len(recent)
        if loss > mean * self.threshold + 1e-3:
            return True
        return False
