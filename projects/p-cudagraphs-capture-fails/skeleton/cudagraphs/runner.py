class CUDAGraphRunner:
    def __init__(self, model):
        raise NotImplementedError

    def capture(self, x):
        raise NotImplementedError

    def run(self, x):
        raise NotImplementedError
