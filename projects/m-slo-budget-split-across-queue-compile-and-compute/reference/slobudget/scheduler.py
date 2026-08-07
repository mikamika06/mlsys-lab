class SLOBudgetScheduler:
    """Schedules requests by dropping those that cannot complete within the SLO budget."""

    def __init__(self, planner):
        self.planner = planner

    def filter_batch(self, batch, current_time_ms):
        admitted = []
        rejected = []
        for req in batch:
            plan = self.planner.compute_budget(
                req["arrival_time_ms"],
                current_time_ms,
                req["shape_key"],
                req["estimated_compute_ms"],
            )
            if plan["is_feasible"]:
                admitted.append({**req, "plan": plan})
            else:
                rejected.append({**req, "plan": plan})
        return admitted, rejected

    def simulate_pipeline(self, requests, compute_cost_fn):
        clock = 0.0
        completed = []
        dropped = []
        wasted_compute = 0.0

        for req in requests:
            arrival = req["arrival_time_ms"]
            if clock < arrival:
                clock = arrival

            plan = self.planner.compute_budget(
                arrival,
                clock,
                req["shape_key"],
                req["estimated_compute_ms"],
            )

            if not plan["is_feasible"]:
                dropped.append(req)
                continue

            compile_time = plan["compile_time"]
            clock += compile_time
            if compile_time > 0:
                self.planner.register_compile_time(req["shape_key"], compile_time)

            compute_time = compute_cost_fn(req)
            total_latency = clock + compute_time - arrival

            if total_latency <= self.planner.slo_ms:
                clock += compute_time
                completed.append(req)
            else:
                clock += compute_time
                wasted_compute += compute_time
                dropped.append(req)

        return {
            "completed": completed,
            "dropped": dropped,
            "wasted_compute_ms": wasted_compute,
        }
