class GraphReducer:
    def __init__(self, nodes):
        self.nodes = nodes

    def step_compare(self, ref_outputs, test_outputs):
        raise NotImplementedError

    def bisection(self, inputs, oracle_fn, test_fn):
        raise NotImplementedError

    def isolate(self, node_id, input_data):
        raise NotImplementedError

    def patch(self, node_id, replacement):
        raise NotImplementedError

    def verify(self, inputs, expected):
        raise NotImplementedError
