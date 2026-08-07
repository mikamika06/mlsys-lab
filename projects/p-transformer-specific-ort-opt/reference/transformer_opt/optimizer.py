import copy
import numpy as np

class TransformerOptimizer:
    def __init__(self, model):
        self.model = copy.deepcopy(model)

    def optimize(self, num_heads=4, hidden_size=64):
        nodes = self.model["nodes"]
        new_nodes = []
        i = 0
        fused_count = 0

        while i < len(nodes):
            if (i + 4 < len(nodes) and
                nodes[i]["op"] == "MatMul" and
                nodes[i+1]["op"] == "Div" and
                nodes[i+2]["op"] == "Softmax" and
                nodes[i+3]["op"] == "MatMul"):

                q_in = nodes[i]["inputs"][0]
                k_in = nodes[i]["inputs"][1]
                v_in = nodes[i+3]["inputs"][1]
                out = nodes[i+3]["outputs"][0]

                fused_node = {
                    "op": "Attention",
                    "inputs": [q_in, k_in, v_in],
                    "outputs": [out],
                    "attributes": {
                        "num_heads": num_heads,
                        "hidden_size": hidden_size,
                        "scale": 1.0 / np.sqrt(hidden_size // num_heads)
                    }
                }
                new_nodes.append(fused_node)
                i += 4
                fused_count += 1
            elif (i + 2 < len(nodes) and
                  nodes[i]["op"] == "ReduceMean" and
                  nodes[i+1]["op"] == "Sub" and
                  nodes[i+2]["op"] == "Div"):

                inp = nodes[i]["inputs"][0]
                out = nodes[i+2]["outputs"][0]
                fused_node = {
                    "op": "LayerNormalization",
                    "inputs": [inp],
                    "outputs": [out],
                    "attributes": {"axis": -1, "epsilon": 1e-5}
                }
                new_nodes.append(fused_node)
                i += 3
                fused_count += 1
            else:
                new_nodes.append(nodes[i])
                i += 1

        opt_model = copy.deepcopy(self.model)
        opt_model["nodes"] = new_nodes
        opt_model["is_optimized"] = True
        opt_model["fused_count"] = fused_count
        return opt_model
