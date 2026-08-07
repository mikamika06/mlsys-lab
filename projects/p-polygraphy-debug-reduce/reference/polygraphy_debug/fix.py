import numpy as np


class GraphPatcher:
    def __init__(self, execution_graph):
        self.graph = execution_graph

    def patch_node(self, node_id, replacement_op):
        self.graph.replace_op(node_id, replacement_op)

    def verify_remediation(self, ref_runner, target_runner, inputs, tolerance=1e-3):
        ref_out = ref_runner.run_full(self.graph, inputs)
        tgt_out = target_runner.run_full(self.graph, inputs)
        diff = float(np.max(np.abs(ref_out - tgt_out)))
        return {"parity_restored": diff <= tolerance, "max_mae": diff}
