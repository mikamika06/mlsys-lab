class Engine:
    def __init__(self, mps_kernels=None):
        raise NotImplementedError

    def list_unimplemented_ops(self, graph):
        raise NotImplementedError

    def fallback_share(self, trace):
        raise NotImplementedError

    def rewrite_op(self, op_name, new_fn):
        raise NotImplementedError

    def run(self, graph):
        raise NotImplementedError

    def hot_path_fallbacks(self, graph):
        raise NotImplementedError
