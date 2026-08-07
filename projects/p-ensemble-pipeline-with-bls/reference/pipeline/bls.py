class BLSOrchestrator:
    def __init__(self, ensemble_dag):
        if not ensemble_dag.validate():
            raise ValueError("Invalid DAG configuration")
        self.dag = ensemble_dag

    def execute_in_process(self, initial_input):
        computed = {}
        bytes_transferred = 0

        for stage_name, stage in self.dag.stages.items():
            deps = self.dag.dependencies[stage_name]
            if not deps:
                inp = initial_input
            else:
                inp = computed[deps[0]]

            out = stage.run(inp)
            computed[stage_name] = out

        last_stage = list(self.dag.stages.keys())[-1]
        return {
            "output": computed[last_stage],
            "bytes_transferred": bytes_transferred,
            "latency_ms": 0.1,
        }

    def measure_overhead(self, initial_input, remote_latency_ms=5.0):
        remote_res = self.dag.execute_remote(
            initial_input, network_latency_ms=remote_latency_ms
        )
        bls_res = self.execute_in_process(initial_input)

        remote_bytes = remote_res["bytes_transferred"]
        bls_bytes = bls_res["bytes_transferred"]
        saved_bytes = max(0, remote_bytes - bls_bytes)
        bytes_saved_ratio = saved_bytes / max(1, remote_bytes)

        speedup = remote_res["latency_ms"] / max(0.001, bls_res["latency_ms"])

        return {
            "remote_bytes": remote_bytes,
            "bls_bytes": bls_bytes,
            "bytes_saved_ratio": bytes_saved_ratio,
            "remote_latency": remote_res["latency_ms"],
            "bls_latency": bls_res["latency_ms"],
            "speedup": speedup,
        }

    def execute_with_fault_tolerance(self, initial_input, fallback_responses=None):
        if fallback_responses is None:
            fallback_responses = {}

        computed = {}
        for stage_name, stage in self.dag.stages.items():
            deps = self.dag.dependencies[stage_name]
            if not deps:
                inp = initial_input
            else:
                inp = computed[deps[0]]

            try:
                out = stage.run(inp)
            except Exception as e:
                if stage_name in fallback_responses:
                    out = fallback_responses[stage_name]
                else:
                    raise RuntimeError(
                        f"Stage {stage_name} failed without fallback: {e}"
                    ) from e

            computed[stage_name] = out

        last_stage = list(self.dag.stages.keys())[-1]
        return computed[last_stage]
