class DependencyGraph:
    def __init__(self):
        raise NotImplementedError

    def add_node(self, name, shape):
        raise NotImplementedError

    def add_edge(self, src, dst, dim_src, dim_dst):
        raise NotImplementedError
