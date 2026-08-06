class KernelAttributionHarness:
    def __init__(self):
        raise NotImplementedError

    def register_trace(self, trace_events):
        raise NotImplementedError

    def attribute_kernels(self):
        raise NotImplementedError
