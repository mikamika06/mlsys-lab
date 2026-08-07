import numpy as np

class GraphReducer:
    def __init__(self, nodes):
        self.nodes = nodes

    def step_compare(self, ref_outputs, test_outputs):
        mismatches = []
        for k in ref_outputs:
            if k not in test_outputs:
                mismatches.append(k)
                continue
            ref_val = np.asarray(ref_outputs[k])
            test_val = np.asarray(test_outputs[k])
            if not np.allclose(ref_val, test_val, atol=1e-3, rtol=1e-3):
                mismatches.append(k)
        return mismatches

    def bisection(self, inputs, oracle_fn, test_fn):
        low = 0
        high = len(self.nodes) - 1
        faulty_idx = -1
        while low <= high:
            mid = (low + high) // 2
            sub_nodes = self.nodes[:mid+1]
            ref_out = oracle_fn(sub_nodes, inputs)
            test_out = test_fn(sub_nodes, inputs)
            if not np.allclose(ref_out, test_out, atol=1e-3, rtol=1e-3):
                faulty_idx = mid
                high = mid - 1
            else:
                low = mid + 1
        return faulty_idx

    def isolate(self, node_id, input_data):
        node = self.nodes[node_id]
        return node(input_data)

    def patch(self, node_id, replacement):
        self.nodes[node_id] = replacement

    def verify(self, inputs, expected):
        curr = inputs
        for n in self.nodes:
            curr = n(curr)
        return np.allclose(curr, expected, atol=1e-3, rtol=1e-3)
