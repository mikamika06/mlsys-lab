class StaticBufferHarness:
    """Static buffer harness managing variable input sizes for captured graph executions."""

    def __init__(self, max_shape, dtype=None):
        raise NotImplementedError

    def update_input(self, tensor):
        raise NotImplementedError

    def run(self, graph_runner):
        raise NotImplementedError
