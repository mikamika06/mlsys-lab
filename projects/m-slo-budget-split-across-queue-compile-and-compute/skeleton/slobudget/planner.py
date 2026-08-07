class BudgetPlanner:
    """Calculates remaining SLO compute budget given queue delay and JIT compile overhead."""

    def __init__(self, slo_ms, compilation_cache=None):
        raise NotImplementedError

    def predict_compile_time(self, shape_key):
        raise NotImplementedError

    def register_compile_time(self, shape_key, duration_ms):
        raise NotImplementedError

    def compute_budget(self, arrival_time_ms, current_time_ms, shape_key, estimated_compute_ms):
        raise NotImplementedError
