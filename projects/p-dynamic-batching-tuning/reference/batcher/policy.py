class BatchingPolicy:
    def __init__(self, config):
        self.config = config
        self.small_queue = []
        self.large_queue = []

    def add_request(self, req):
        if self.config.get("split_queues", False):
            if req.get("tokens", 0) <= 128:
                self.small_queue.append(req)
            else:
                self.large_queue.append(req)
        else:
            self.small_queue.append(req)

    def decide(self, current_time):
        batches = []
        max_bs = self.config["max_batch_size"]
        timeout = self.config["timeout_ms"]

        for q in [self.small_queue, self.large_queue]:
            if not q:
                continue
            oldest_age = current_time - q[0]["arrival"]
            if len(q) >= max_bs or oldest_age >= timeout:
                batch_items = q[:max_bs]
                del q[:len(batch_items)]
                batches.append(batch_items)
        return batches
