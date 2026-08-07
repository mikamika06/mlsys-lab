class TieredQueueManager:
    def __init__(self, thresholds):
        self.thresholds = sorted(thresholds)
        self.queues = {i: [] for i in range(len(self.thresholds) + 1)}

    def push(self, request):
        size = request.get("size", 1)
        placed = False
        for idx, t in enumerate(self.thresholds):
            if size <= t:
                self.queues[idx].append(request)
                placed = True
                break
        if not placed:
            self.queues[len(self.thresholds)].append(request)

    def pop_batch(self, max_size):
        batch = []
        for q_idx in sorted(self.queues.keys()):
            while self.queues[q_idx] and len(batch) < max_size:
                batch.append(self.queues[q_idx].pop(0))
        return batch
