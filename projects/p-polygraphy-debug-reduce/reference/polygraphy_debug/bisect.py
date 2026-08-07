import numpy as np


class GraphBisector:
    def __init__(self, execution_graph, ref_runner, target_runner, tolerance=1e-3):
        self.graph = execution_graph
        self.ref_runner = ref_runner
        self.target_runner = target_runner
        self.tolerance = tolerance
        self.steps_taken = 0

    def bisect(self, inputs):
        nodes = self.graph.topological_order()
        low = 0
        high = len(nodes) - 1
        culprit = None
        self.steps_taken = 0

        while low <= high:
            self.steps_taken += 1
            mid = (low + high) // 2
            prefix_nodes = nodes[: mid + 1]

            state = dict(inputs)
            for nid in prefix_nodes:
                state[nid] = self.ref_runner.run_node(nid, state)

            test_node = nodes[mid]
            ref_out = state[test_node]
            tgt_out = self.target_runner.run_node(test_node, state)

            diff = float(np.max(np.abs(ref_out - tgt_out)))
            if diff > self.tolerance:
                culprit = test_node
                high = mid - 1
            else:
                low = mid + 1

        return {"culprit": culprit, "steps": self.steps_taken}
