class BackendRegistry:
    def __init__(self):
        self._backends = {}

    def register(self, name, backend_cls):
        self._backends[name] = backend_cls

    def get(self, name):
        return self._backends.get(name)

class CompilationGraph:
    def __init__(self, nodes):
        self.nodes = list(nodes)

    def get_nodes(self):
        return self.nodes
