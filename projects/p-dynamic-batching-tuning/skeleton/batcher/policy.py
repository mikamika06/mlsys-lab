class BatchingPolicy:
    def __init__(self, config):
        raise NotImplementedError

    def decide(self, queue, current_time):
        raise NotImplementedError
