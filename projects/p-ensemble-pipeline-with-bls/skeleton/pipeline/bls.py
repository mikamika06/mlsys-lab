class BLSOrchestrator:
    def __init__(self, ensemble_dag):
        raise NotImplementedError

    def execute_in_process(self, initial_input):
        raise NotImplementedError

    def measure_overhead(self, initial_input, remote_latency_ms=5.0):
        raise NotImplementedError

    def execute_with_fault_tolerance(self, initial_input, fallback_responses=None):
        raise NotImplementedError
