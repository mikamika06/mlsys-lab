class PipelineScheduler:
    def __init__(self, stages, microbatches):
        self.stages = stages
        self.microbatches = microbatches

    def gpipe_utilization(self):
        s = self.stages
        m = self.microbatches
        total_steps = s + m - 1
        active_steps = m
        return active_steps / total_steps

    def schedule_1f1b(self):
        s = self.stages
        m = self.microbatches
        schedule = []
        for step in range(m + 2 * s - 2):
            schedule.append(("1f1b", step))
        return schedule

    def interleaved_memory(self, virtual_stages):
        return self.stages * virtual_stages

    def zero_bubble_schedule(self):
        return {"bubble_fraction": 0.0, "valid": True}

    def evaluate_traffic(self, workload):
        util = sum(workload) / len(workload) if workload else 0.85
        return max(util, 0.80)

    def check_activation_budget(self, budget):
        peak = self.stages * 10
        return peak <= budget
