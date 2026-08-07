class AdmissionPolicy:
    def __init__(self, slo_target, max_queue_cost=100.0):
        raise NotImplementedError

    def should_admit(self, request, queue_model):
        raise NotImplementedError
