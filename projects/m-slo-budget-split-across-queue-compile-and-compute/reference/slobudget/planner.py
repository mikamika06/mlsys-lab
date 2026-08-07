class BudgetPlanner:
    """Calculates remaining SLO compute budget given queue delay and JIT compile overhead."""

    def __init__(self, slo_ms, compilation_cache=None):
        self.slo_ms = float(slo_ms)
        self.compilation_cache = dict(compilation_cache) if compilation_cache else {}

    def predict_compile_time(self, shape_key):
        if shape_key in self.compilation_cache:
            return 0.0
        return 15.0

    def register_compile_time(self, shape_key, duration_ms):
        self.compilation_cache[shape_key] = float(duration_ms)

    def compute_budget(self, arrival_time_ms, current_time_ms, shape_key, estimated_compute_ms):
        queue_delay = current_time_ms - arrival_time_ms
        compile_time = self.predict_compile_time(shape_key)
        remaining = self.slo_ms - queue_delay - compile_time
        return {
            "queue_delay": queue_delay,
            "compile_time": compile_time,
            "remaining_compute_budget": remaining,
            "is_feasible": remaining >= estimated_compute_ms and remaining > 0,
        }
