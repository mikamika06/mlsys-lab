class ChunkedScheduler:
    def __init__(self, config):
        raise NotImplementedError

    def add_request(self, req):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def metrics(self):
        raise NotImplementedError
