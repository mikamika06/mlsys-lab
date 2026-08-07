class BackendRegistry:
    def __init__(self):
        raise NotImplementedError

    def register(self, name, backend_cls):
        raise NotImplementedError

    def get(self, name):
        raise NotImplementedError

class CompilationGraph:
    def __init__(self, nodes):
        raise NotImplementedError

    def get_nodes(self):
        raise NotImplementedError
