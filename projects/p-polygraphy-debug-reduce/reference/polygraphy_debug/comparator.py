import numpy as np


class StepComparator:
    def __init__(self, ref_runner, target_runner, rtol=1e-3, atol=1e-4):
        self.ref_runner = ref_runner
        self.target_runner = target_runner
        self.rtol = rtol
        self.atol = atol

    def compare_layer(self, layer_name, inputs):
        ref_out = self.ref_runner.run_node(layer_name, inputs)
        tgt_out = self.target_runner.run_node(layer_name, inputs)
        diff = np.abs(ref_out - tgt_out)
        max_mae = float(np.max(diff))
        match = bool(np.allclose(ref_out, tgt_out, rtol=self.rtol, atol=self.atol))
        return {"match": match, "max_mae": max_mae, "ref": ref_out, "target": tgt_out}

    def find_first_divergence(self, execution_graph, inputs):
        curr_inputs = dict(inputs)
        for node_id in execution_graph.topological_order():
            ref_out = self.ref_runner.run_node(node_id, curr_inputs)
            tgt_out = self.target_runner.run_node(node_id, curr_inputs)
            diff = np.abs(ref_out - tgt_out)
            max_mae = float(np.max(diff))
            match = np.allclose(ref_out, tgt_out, rtol=self.rtol, atol=self.atol)
            if not match:
                return {"diverged": True, "node_id": node_id, "max_mae": max_mae}
            curr_inputs[node_id] = ref_out
        return {"diverged": False, "node_id": None, "max_mae": 0.0}
