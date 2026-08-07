from preempt.policy import PreemptionPolicy

class PreemptionScheduler:
    def __init__(self, config, policy=None):
        self.config = config
        self.policy = policy if policy else PreemptionPolicy(config)
        self.waiting = []
        self.running = []
        self.swapped = []
        self.max_running = config.get("max_running", 4)
        self.time = 0
        self.latencies = []

    def add_requests(self, requests):
        for r in requests:
            r["arrival_time"] = self.time
            r["context_len"] = len(r.get("prompt", []))
            r["processed_tokens"] = 0
            self.waiting.append(r)

    def step(self):
        self.time += 1
        while self.waiting and len(self.running) < self.max_running:
            req = self.waiting.pop(0)
            self.running.append(req)

        if len(self.running) > self.max_running and self.waiting:
            victim = self.running.pop(0)
            decision = self.policy.decide(victim, {})
            victim["preemption_mode"] = decision
            if decision == "swap":
                self.swapped.append(victim)
            else:
                self.waiting.insert(0, victim)

        for req in list(self.running):
            req["processed_tokens"] += 1
            if req["processed_tokens"] >= req.get("total_len", 32):
                latency = self.time - req["arrival_time"]
                self.latencies.append(latency)
                self.running.remove(req)

    def run_trace(self, trace):
        for t, reqs in sorted(trace.items()):
            while self.time < t:
                self.step()
            self.add_requests(reqs)
        while self.running or self.waiting or self.swapped:
            self.step()
        return self.latencies
