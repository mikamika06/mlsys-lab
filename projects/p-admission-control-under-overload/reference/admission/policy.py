class AdmissionPolicy:
    def __init__(self, slo_target, max_queue_cost=100.0):
        self.slo_target = slo_target
        self.max_queue_cost = max_queue_cost

    def should_admit(self, request, queue_model):
        current_cost = sum(r.cost for r in queue_model.queue)
        if current_cost + request.cost > self.max_queue_cost:
            return False
        est_wait = (current_cost / queue_model.processing_rate) if queue_model.processing_rate > 0 else 0.0
        total_est = est_wait + (request.cost / queue_model.processing_rate)
        if total_est > self.slo_target:
            if request.priority < 2:
                return False
        return True
