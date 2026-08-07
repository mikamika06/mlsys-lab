class TierManager:
    def __init__(self, gpu_capacity, cpu_capacity):
        self.gpu_capacity = gpu_capacity
        self.cpu_capacity = cpu_capacity
        self.gpu_used = 0
        self.cpu_used = 0
        self.sessions = {}
        self.latencies = []

    def register_session(self, session_id, size, last_access=0):
        self.sessions[session_id] = {"size": size, "location": "gpu", "last_access": last_access}
        self.gpu_used += size

    def offload(self, session_id):
        if session_id in self.sessions and self.sessions[session_id]["location"] == "gpu":
            sz = self.sessions[session_id]["size"]
            self.gpu_used -= sz
            self.cpu_used += sz
            self.sessions[session_id]["location"] = "cpu"
            return True
        return False

    def prefetch(self, session_id):
        if session_id in self.sessions and self.sessions[session_id]["location"] == "cpu":
            sz = self.sessions[session_id]["size"]
            self.cpu_used -= sz
            self.gpu_used += sz
            self.sessions[session_id]["location"] = "gpu"
            self.latencies.append(15.0)
            return True
        return False

    def record_latency(self, val):
        self.latencies.append(val)

    def get_p95_latency(self):
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        idx = int(0.95 * len(sorted_l))
        if idx >= len(sorted_l):
            idx = len(sorted_l) - 1
        return sorted_l[idx]
