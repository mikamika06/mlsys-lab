import numpy as np

class GraphAnalyzer:
    def __init__(self, original_model, optimized_model):
        self.original_model = original_model
        self.optimized_model = optimized_model

    def analyze_fused_nodes(self):
        orig_counts = {}
        for n in self.original_model["nodes"]:
            op = n["op"]
            orig_counts[op] = orig_counts.get(op, 0) + 1

        opt_counts = {}
        for n in self.optimized_model["nodes"]:
            op = n["op"]
            opt_counts[op] = opt_counts.get(op, 0) + 1

        removed_ops = {}
        for op, count in orig_counts.items():
            diff = count - opt_counts.get(op, 0)
            if diff > 0:
                removed_ops[op] = diff

        added_ops = {}
        for op, count in opt_counts.items():
            diff = count - orig_counts.get(op, 0)
            if diff > 0:
                added_ops[op] = diff

        return {
            "original_node_count": len(self.original_model["nodes"]),
            "optimized_node_count": len(self.optimized_model["nodes"]),
            "node_reduction": len(self.original_model["nodes"]) - len(self.optimized_model["nodes"]),
            "removed_ops": removed_ops,
            "added_ops": added_ops,
            "fused_attention_count": opt_counts.get("Attention", 0),
            "fused_layernorm_count": opt_counts.get("LayerNormalization", 0)
        }

    def check_parity(self, input_data, rtol=1e-3, atol=1e-4):
        from ref import run_model

        orig_out = run_model(self.original_model, input_data)
        opt_out = run_model(self.optimized_model, input_data)

        max_diff = float(np.max(np.abs(orig_out - opt_out)))
        is_close = bool(np.allclose(orig_out, opt_out, rtol=rtol, atol=atol))

        return {
            "parity_match": is_close,
            "max_absolute_difference": max_diff,
            "rtol": rtol,
            "atol": atol
        }
