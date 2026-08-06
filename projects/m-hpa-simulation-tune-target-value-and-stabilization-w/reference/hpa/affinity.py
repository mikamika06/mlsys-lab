import random


def evaluate_session_affinity(sessions, num_replicas, routing_strategy):
    hits = 0
    total_turns = 0

    for session in sessions:
        last_replica = None
        for turn in session:
            total_turns += 1
            if routing_strategy == "random":
                replica = random.randint(0, num_replicas - 1)
            elif routing_strategy == "affinity":
                replica = session[0] % num_replicas
            else:
                replica = 0

            if last_replica is not None and replica == last_replica:
                hits += 1
            last_replica = replica

    hit_rate = hits / total_turns if total_turns > 0 else 0.0
    return hit_rate
