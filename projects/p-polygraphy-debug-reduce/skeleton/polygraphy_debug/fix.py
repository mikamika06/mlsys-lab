class GraphPatcher:
    def __init__(self, execution_graph):
        raise NotImplementedError

    def patch_node(self, node_id, replacement_op):
        raise NotImplementedError

    def verify_remediation(self, ref_runner, target_runner, inputs, tolerance=1e-3):
        raise NotImplementedError
