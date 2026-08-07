class Autotuner:
    def __init__(self, configs):
        raise NotImplementedError
    def benchmark(self, fn, args):
        raise NotImplementedError
    def select(self, shapes, strides, work_fn):
        raise NotImplementedError
