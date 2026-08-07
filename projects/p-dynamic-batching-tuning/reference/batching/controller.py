class DynamicBatchController:
    def __init__(self, profile, slo):
        self.profile = profile
        self.slo = slo
        self.queue = []

    def step(self, incoming, current_load):
        for req in incoming:
            self.queue.append(req)

        batch = []
        if len(self.queue) >= 4 or current_load > 0.8:
            batch = self.queue[:8]
            self.queue = self.queue[8:]
        elif len(self.queue) > 0 and current_load < 0.3:
            batch = self.queue[:4]
            self.queue = self.queue[4:]
        return batch
