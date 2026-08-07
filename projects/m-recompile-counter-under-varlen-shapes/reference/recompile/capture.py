def capture_decode_step(func, sample_inputs):
    """Capture a decode step."""
    class MockGraph:
        def __init__(self, inputs):
            self.inputs = inputs
            self.captured = True

        def replay(self):
            return func(*self.inputs)

    return MockGraph(sample_inputs)
