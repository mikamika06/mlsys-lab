from runner.engine import Engine, Request, RequestMetrics

class LoadBench:
    def __init__(self, warmup_runs: int = 2):
        self.warmup_runs = warmup_runs

    def generate_workload(self, num_users: int, prompt_len: int = 32, output_len: int = 100) -> list[Request]:
        return [
            Request(
                req_id=f"req_{i}",
                arrival_time=0.0,
                prompt_len=prompt_len,
                output_len=output_len
            )
            for i in range(num_users)
        ]

    def run_benchmark(self, engine: Engine, workload: list[Request]) -> dict:
        if self.warmup_runs > 0:
            warmup_reqs = [
                Request(
                    req_id=f"warmup_{i}",
                    arrival_time=0.0,
                    prompt_len=16,
                    output_len=10
                )
                for i in range(self.warmup_runs)
            ]
            engine.run_trace(warmup_reqs)

        metrics = engine.run_trace(workload)

        if not metrics:
            return {
                "warmup_completed": True if self.warmup_runs > 0 else False,
                "num_users": len(workload),
                "total_tokens": 0,
                "total_wall_time_ms": 0.0,
                "aggregate_tok_per_sec": 0.0,
                "mean_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "metrics": []
            }

        total_tokens = sum(m.tokens_generated for m in metrics)
        min_start = min(m.start_time for m in metrics)
        max_finish = max(m.finish_time for m in metrics)
        wall_time_ms = max_finish - min_start

        agg_tps = (total_tokens / (wall_time_ms / 1000.0)) if wall_time_ms > 0 else 0.0
        latencies = [m.total_time_ms for m in metrics]
        mean_lat = sum(latencies) / len(latencies)

        sorted_lat = sorted(latencies)
        p95_idx = int(0.95 * len(sorted_lat))
        if p95_idx >= len(sorted_lat):
            p95_idx = len(sorted_lat) - 1
        p95_lat = sorted_lat[p95_idx]

        return {
            "warmup_completed": True if self.warmup_runs > 0 else False,
            "num_users": len(workload),
            "total_tokens": total_tokens,
            "total_wall_time_ms": wall_time_ms,
            "aggregate_tok_per_sec": float(agg_tps),
            "mean_latency_ms": float(mean_lat),
            "p95_latency_ms": float(p95_lat),
            "metrics": metrics
        }
