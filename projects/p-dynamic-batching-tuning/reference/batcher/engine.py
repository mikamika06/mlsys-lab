class BatchEngine:
    def __init__(self, policy):
        self.policy = policy

    def run(self, trace):
        time = 0.0
        results = []
        trace_idx = 0

        while trace_idx < len(trace) or self.policy.small_queue or self.policy.large_queue:
            while trace_idx < len(trace) and trace[trace_idx]["arrival"] <= time:
                self.policy.add_request(trace[trace_idx])
                trace_idx += 1

            batches = self.policy.decide(time)
            for b in batches:
                exec_time = 5.0 + len(b) * 0.5
                for req in b:
                    results.append({
                        "rid": req["rid"],
                        "latency": time + exec_time - req["arrival"]
                    })
            time += 1.0
        return results
