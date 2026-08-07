class PipelineScheduler:
    def __init__(self, stages, microbatches):
        raise NotImplementedError

    def gpipe_utilization(self):
        raise NotImplementedError

    def schedule_1f1b(self):
        raise NotImplementedError

    def interleaved_memory(self, virtual_stages):
        raise NotImplementedError

    def zero_bubble_schedule(self):
        raise NotImplementedError

    def evaluate_traffic(self, workload):
        raise NotImplementedError

    def check_activation_budget(self, budget):
        raise NotImplementedError
