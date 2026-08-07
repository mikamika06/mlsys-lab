class CompilerEngine:
    def __init__(self):
        raise NotImplementedError

    def trace_ops(self, graph):
        raise NotImplementedError

    def compile_and_run(self, op):
        raise NotImplementedError

    def export_cache(self, path):
        raise NotImplementedError

    def import_cache(self, path):
        raise NotImplementedError

    def warmup(self, graphs):
        raise NotImplementedError

    def invalidate(self, new_version):
        raise NotImplementedError
