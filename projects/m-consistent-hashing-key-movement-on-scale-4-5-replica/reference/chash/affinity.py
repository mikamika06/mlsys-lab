class SessionAffinityRouter:
    def __init__(self, ring, ttl_seconds):
        """Router managing sticky sessions with key TTL expiration."""
        self.ring = ring
        self.ttl_seconds = ttl_seconds
        self.sessions = {}

    def route(self, session_id, key, current_time):
        if session_id in self.sessions:
            replica, last_time = self.sessions[session_id]
            if current_time - last_time <= self.ttl_seconds:
                self.sessions[session_id] = (replica, current_time)
                return replica

        replica = self.ring.get_replica(key)
        self.sessions[session_id] = (replica, current_time)
        return replica

    def evaluate_affinity_ttl(self, sessions, total_duration, key_churn_rate):
        """Simulates TTL effects and returns score based on hit rate and balance."""
        best_ttl = self.ttl_seconds
        best_score = -1.0
        candidate_ttls = [5, 10, 30, 60, 120, 300, 600]

        for ttl in candidate_ttls:
            self.ttl_seconds = ttl
            self.sessions.clear()
            hits = 0
            total_requests = 0
            replica_counts = {}

            for t in range(total_duration):
                for sess_id, keys in sessions.items():
                    key_idx = int((t * key_churn_rate) % len(keys))
                    key = keys[key_idx]

                    if sess_id in self.sessions:
                        prev_rep, prev_t = self.sessions[sess_id]
                        if t - prev_t <= ttl and self.ring.get_replica(key) == prev_rep:
                            hits += 1

                    rep = self.route(sess_id, key, t)
                    replica_counts[rep] = replica_counts.get(rep, 0) + 1
                    total_requests += 1

            hit_rate = hits / total_requests if total_requests > 0 else 0
            counts = list(replica_counts.values())
            imbalance = (max(counts) - min(counts)) / (sum(counts) + 1e-5) if counts else 1.0

            score = hit_rate - (0.5 * imbalance)
            if score > best_score:
                best_score = score
                best_ttl = ttl

        self.ttl_seconds = best_ttl
        return {"optimal_ttl": best_ttl, "score": float(best_score)}
