class TieredQueueManager:
    def __init__(self, thresholds):
        raise NotImplementedError
    def push(self, request):
        raise NotImplementedError
    def pop_batch(self, max_size):
        raise NotImplementedError
