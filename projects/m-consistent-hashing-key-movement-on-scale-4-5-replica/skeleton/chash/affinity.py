class SessionAffinityRouter:
    def __init__(self, ring, ttl_seconds):
        raise NotImplementedError

    def route(self, session_id, key, current_time):
        raise NotImplementedError

    def evaluate_affinity_ttl(self, sessions, total_duration, key_churn_rate):
        raise NotImplementedError
