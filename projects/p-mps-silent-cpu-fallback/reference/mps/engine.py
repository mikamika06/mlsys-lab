import numpy as np


class Engine:
    def __init__(self, mps_kernels=None):
        self.mps_kernels = mps_kernels or set(["add", "mul", "relu", "matmul"])
        self.rewrites = {}

    def list_unimplemented_ops(self, graph):
        unimplemented = []
        for op in graph:
            name = op["name"]
            if name not in self.mps_kernels and name not in self.rewrites:
                if name not in unimplemented:
                    unimplemented.append(name)
        return sorted(unimplemented)

    def fallback_share(self, trace):
        total_time = 0.0
        fallback_time = 0.0
        for step in trace:
            t = step.get("duration", 1.0)
            total_time += t
            if step.get("fallback", False):
                fallback_time += t
        if total_time == 0.0:
            return 0.0
        return fallback_time / total_time

    def rewrite_op(self, op_name, new_fn):
        self.rewrites[op_name] = new_fn

    def run(self, graph):
        trace = []
        for op in graph:
            name = op["name"]
            is_fallback = False
            if name in self.rewrites:
                duration = op.get("base_duration", 1.0) * 0.5
            elif name in self.mps_kernels:
                duration = op.get("base_duration", 1.0) * 0.2
            else:
                is_fallback = True
                duration = op.get("base_duration", 1.0) * 2.0
            trace.append({"name": name, "duration": duration, "fallback": is_fallback})
        return trace

    def hot_path_fallbacks(self, graph):
        trace = self.run(graph)
        fallbacks = [step["name"] for step in trace if step["fallback"]]
        return len(fallbacks)
