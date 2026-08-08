def oracle_transfer_cost(num_tokens, bytes_per_token, bandwidth_gbps):
    if bandwidth_gbps <= 0:
        return float("inf")
    return (num_tokens * bytes_per_token) / (bandwidth_gbps * 1024.0 * 1024.0 * 1024.0)


def oracle_select_offload(sessions, gpu_budget):
    sorted_sessions = sorted(sessions, key=lambda s: (s.get("priority", 0), s.get("last_accessed", 0)))
    offloaded = []
    current_usage = sum(s["tokens"] for s in sessions)
    for s in sorted_sessions:
        if current_usage <= gpu_budget:
            break
        offloaded.append(s["id"])
        current_usage -= s["tokens"]
    return offloaded


def oracle_should_prefetch(session_id, request_queue, cpu_tier):
    if session_id not in cpu_tier:
        return False
    for req in request_queue:
        if req.get("session_id") == session_id:
            return True
    return False


class OracleTierManager:
    def __init__(self, capacity):
        self.capacity = capacity
        self.storage = {}
        self.used = 0

    def offload(self, session_id, blocks):
        if self.used + len(blocks) > self.capacity:
            return False
        self.storage[session_id] = blocks
        self.used += len(blocks)
        return True

    def bring_back(self, session_id):
        if session_id not in self.storage:
            return None
        blocks = self.storage.pop(session_id)
        self.used -= len(blocks)
        return blocks

    def __contains__(self, session_id):
        return session_id in self.storage
