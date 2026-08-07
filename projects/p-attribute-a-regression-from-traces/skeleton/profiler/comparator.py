class TraceComparator:
    def __init__(self, trace_a, trace_b):
        raise NotImplementedError

    def reduce_trace(self, trace):
        raise NotImplementedError

    def find_max_delta(self):
        raise NotImplementedError

    def classify_kernel(self, kernel_name):
        raise NotImplementedError

    def detect_synchronization(self):
        raise NotImplementedError

    def confirm_root_cause(self, trace_c):
        raise NotImplementedError
