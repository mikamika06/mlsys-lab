import numpy as np


class Profiler:
    def __init__(self, flop_cost_per_ms=300000.0):
        self.flop_cost_per_ms = flop_cost_per_ms

    def profile(self, graph, input_tensor):
        shape = input_tensor.shape
        op_profiles = []
        breakdown = {}
        total_flops = 0
        total_latency = 0.0

        curr_shape = shape
        for node in graph.nodes:
            flops = node.compute_flops(curr_shape)
            lat = float(flops) / self.flop_cost_per_ms
            op_profiles.append({
                "node_name": node.name,
                "op_type": node.op_type,
                "flops": flops,
                "latency_ms": lat,
            })
            breakdown[node.op_type] = breakdown.get(node.op_type, 0.0) + lat
            total_flops += flops
            total_latency += lat

            if node.op_type == "conv2d":
                c_out = node.params.get("out_channels", 16)
                curr_shape = (curr_shape[0], c_out, curr_shape[2], curr_shape[3])
            elif node.op_type == "linear":
                out_f = node.params.get("out_features", 10)
                curr_shape = (curr_shape[0], out_f)

        bottleneck = max(op_profiles, key=lambda x: x["latency_ms"])["node_name"] if op_profiles else ""

        return {
            "total_latency_ms": round(total_latency, 2),
            "total_flops": total_flops,
            "op_profiles": op_profiles,
            "breakdown_by_op_type": breakdown,
            "bottleneck_node": bottleneck,
        }
