class DeviceProfiler:
    def __init__(self, config=None):
        raise NotImplementedError

    def run_warmup(self, iterations):
        raise NotImplementedError

    def separate_steady_state(self, samples):
        raise NotImplementedError

    def measure_cold_start(self):
        raise NotImplementedError

    def compute_required_runs(self, variance, target_error):
        raise NotImplementedError

    def check_session_overlap(self, sessions):
        raise NotImplementedError

    def generate_report(self, data):
        raise NotImplementedError
