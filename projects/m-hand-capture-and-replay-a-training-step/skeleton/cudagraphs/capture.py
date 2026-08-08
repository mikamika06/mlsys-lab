class GraphCaptureSimulator:

    def __init__(self, step_trace):
        raise NotImplementedError

    def warmup(self, inputs):
        raise NotImplementedError

    def capture(self, inputs):
        raise NotImplementedError

    def replay(self, new_inputs):
        raise NotImplementedError
