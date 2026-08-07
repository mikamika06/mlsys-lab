class PipelineStage:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def run(self, inputs):
        return self.fn(inputs)


class EnsembleDAG:
    def __init__(self):
        self.stages = {}
        self.dependencies = {}

    def add_stage(self, name, fn, dependencies=None):
        self.stages[name] = PipelineStage(name, fn)
        self.dependencies[name] = list(dependencies) if dependencies else []

    def validate(self):
        if not self.stages:
            return False
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            for dep in self.dependencies.get(node, []):
                if dep not in self.stages:
                    return False
                if dep not in visited:
                    if not dfs(dep):
                        return False
                elif dep in rec_stack:
                    return False
            rec_stack.remove(node)
            return True

        for stage in self.stages:
            if stage not in visited:
                if not dfs(stage):
                    return False
        return True

    def execute_remote(self, initial_input, network_latency_ms=5.0):
        if not self.validate():
            raise ValueError("Invalid DAG")

        computed = {}
        bytes_transferred = 0
        total_time_ms = 0.0

        for stage_name, stage in self.stages.items():
            deps = self.dependencies[stage_name]
            if not deps:
                inp = initial_input
                bytes_transferred += len(str(initial_input).encode("utf-8"))
            else:
                inp = computed[deps[0]]

            total_time_ms += network_latency_ms
            out = stage.run(inp)
            computed[stage_name] = out
            bytes_transferred += len(str(out).encode("utf-8"))

        last_stage = list(self.stages.keys())[-1]
        return {
            "output": computed[last_stage],
            "bytes_transferred": bytes_transferred,
            "latency_ms": total_time_ms,
        }
