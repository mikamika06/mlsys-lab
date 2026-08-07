class GraphTracer:
    def __init__(self):
        pass

    def trace_graph(self, ops):
        nodes = []
        for i, op in enumerate(ops):
            nodes.append({
                "id": i,
                "op_type": op.get("type", "pointwise"),
                "name": op.get("name", f"op_{i}"),
                "shape_sensitive": op.get("shape_sensitive", False),
                "stride_mismatch": op.get("stride_mismatch", False),
            })
        return {"nodes": nodes}

    def inspect_fusions(self, graph, expected_fusions):
        nodes = graph.get("nodes", [])
        matched = 0
        total = len(expected_fusions)
        for expected in expected_fusions:
            exp_names = set(expected)
            for node in nodes:
                fused_group = node.get("fused_group", [])
                if fused_group and exp_names.issubset(set(fused_group)):
                    matched += 1
                    break
        return float(matched / total) if total > 0 else 1.0

    def find_unfused_nodes(self, graph):
        unfused = []
        for node in graph.get("nodes", []):
            if not node.get("fused", False):
                reason = "unknown"
                if node.get("stride_mismatch"):
                    reason = "stride_mismatch"
                elif node.get("shape_sensitive"):
                    reason = "shape_sensitive_barrier"
                elif node.get("op_type") == "reduction":
                    reason = "reduction_barrier"
                unfused.append({"node": node["name"], "reason": reason})
        return unfused
