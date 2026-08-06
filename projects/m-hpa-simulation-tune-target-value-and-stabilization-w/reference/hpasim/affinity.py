import random


def compute_hit_rate_loss(sessions, replicas, random_routing=True):
    total_requests = 0
    hits = 0
    replica_cache = {r: set() for r in range(replicas)}

    for session in sessions:
        assigned_replica = random.randint(0, replicas - 1) if random_routing else (hash(session["id"]) % replicas)
        for req in session["requests"]:
            total_requests += 1
            if req in replica_cache[assigned_replica]:
                hits += 1
            else:
                replica_cache[assigned_replica].add(req)
                if len(replica_cache[assigned_replica]) > 100:
                    replica_cache[assigned_replica].pop()

    random_hits = hits

    total_requests_sticky = 0
    sticky_hits = 0
    replica_cache_sticky = {r: set() for r in range(replicas)}
    for session in sessions:
        assigned_replica = hash(session["id"]) % replicas
        for req in session["requests"]:
            total_requests_sticky += 1
            if req in replica_cache_sticky[assigned_replica]:
                sticky_hits += 1
            else:
                replica_cache_sticky[assigned_replica].add(req)
                if len(replica_cache_sticky[assigned_replica]) > 100:
                    replica_cache_sticky[assigned_replica].pop()

    hit_rate_random = random_hits / max(1, total_requests)
    hit_rate_sticky = sticky_hits / max(1, total_requests_sticky)
    loss = max(0.0, hit_rate_sticky - hit_rate_random)
    return loss
