class AdmissionController:
    def __init__(self, slo_target, processing_rate=10.0, capacity=100):
        raise NotImplementedError

    def process_incoming(self, requests):
        raise NotImplementedError

    def step(self, time_delta=1.0):
        raise NotImplementedError

    def get_p95_latency(self):
        raise NotImplementedError
