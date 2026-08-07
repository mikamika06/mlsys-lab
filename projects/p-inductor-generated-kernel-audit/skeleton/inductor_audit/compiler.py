class InductorAuditCompiler:
    def __init__(self, config=None):
        raise NotImplementedError

    def compile_graph(self, ops, input_shape):
        raise NotImplementedError

    def run_benchmark(self, compiled_kernel, input_shape):
        raise NotImplementedError

    def clear_cache(self):
        raise NotImplementedError
