class Engine:
    def __init__(self, compile_fn):
        raise NotImplementedError

    def process_stream(self, input_stream):
        raise NotImplementedError
