from admission.model import QueueModel, Request
from admission.policy import AdmissionPolicy

class AdmissionController:
    def __init__(self, slo_target, processing_rate=10.0, capacity=100):
        self.model = QueueModel(capacity=capacity, processing_rate=processing_rate)
        self.policy = AdmissionPolicy(slo_target=slo_target)
        self.admitted_count = 0
        self.rejected_count = 0
        self.latencies = []

    def process_incoming(self, requests):
        for r in requests:
            if self.policy.should_admit(r, self.model):
                if self.model.enqueue(r):
                    self.admitted_count += 1
                else:
                    self.rejected_count += 1
            else:
                self.rejected_count += 1

    def step(self, time_delta=1.0):
        processed_tokens = self.model.processing_rate * time_delta
        while processed_tokens > 0 and self.model.queue:
            r = self.model.queue[0]
            if r.cost <= processed_tokens:
                processed_tokens -= r.cost
                self.model.queue.pop(0)
                completion_time = time_delta
                self.latencies.append(completion_time)
            else:
                r.cost -= processed_tokens
                processed_tokens = 0

    def get_p95_latency(self):
        if not self.latencies:
            return 0.0
        sorted_lats = sorted(self.latencies)
        idx = int(0.95 * len(sorted_lats))
        if idx >= len(sorted_lats):
            idx = len(sorted_lats) - 1
        return sorted_lats[idx]
