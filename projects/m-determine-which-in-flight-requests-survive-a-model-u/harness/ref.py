import random

CONFIGS = [
    {
        "unload_mode": "EXPLICIT",
        "grace_period_ms": 0,
        "queue_delay_ms": 15,
        "requests": [
            {"id": "req-101", "stage": "executing", "remaining_ms": 0},
            {"id": "req-102", "stage": "executing", "remaining_ms": 10},
            {"id": "req-103", "stage": "queued", "remaining_ms": 25},
            {"id": "req-104", "stage": "completed", "remaining_ms": 0},
        ],
    },
    {
        "unload_mode": "GRACEFUL",
        "grace_period_ms": 50,
        "queue_delay_ms": 20,
        "requests": [
            {"id": "req-201", "stage": "executing", "remaining_ms": 30},
            {"id": "req-202", "stage": "executing", "remaining_ms": 60},
            {"id": "req-203", "stage": "queued", "remaining_ms": 25},
            {"id": "req-204", "stage": "queued", "remaining_ms": 35},
        ],
    },
    {
        "unload_mode": "GRACEFUL",
        "grace_period_ms": 20,
        "queue_delay_ms": 10,
        "requests": [
            {"id": "req-301", "stage": "executing", "remaining_ms": 15},
            {"id": "req-302", "stage": "queued", "remaining_ms": 15},
            {"id": "req-303", "stage": "completed", "remaining_ms": 0},
        ],
    },
    {
        "unload_mode": "EXPLICIT",
        "grace_period_ms": 100,
        "queue_delay_ms": 30,
        "requests": [
            {"id": "req-401", "stage": "completed", "remaining_ms": 0},
            {"id": "req-402", "stage": "queued", "remaining_ms": 5},
            {"id": "req-403", "stage": "executing", "remaining_ms": 12},
        ],
    },
]


def generate_synthetic_configs(seed=1337, count=10):
    rng = random.Random(seed)
    out = []
    for i in range(count):
        mode = rng.choice(["EXPLICIT", "GRACEFUL"])
        grace = rng.randint(10, 100) if mode == "GRACEFUL" else 0
        queue_delay = rng.randint(5, 40)
        num_reqs = rng.randint(3, 8)
        reqs = []
        for r_idx in range(num_reqs):
            st = rng.choice(["completed", "executing", "queued"])
            rem = 0 if st == "completed" else rng.randint(5, 80)
            reqs.append({
                "id": f"req-gen-{i}-{r_idx}",
                "stage": st,
                "remaining_ms": rem,
            })
        out.append({
            "unload_mode": mode,
            "grace_period_ms": grace,
            "queue_delay_ms": queue_delay,
            "requests": reqs,
        })
    return out


def determine_surviving_requests(config):
    mode = config.get("unload_mode", "EXPLICIT")
    grace = config.get("grace_period_ms", 0) if mode == "GRACEFUL" else 0
    survivors = []

    for req in config.get("requests", []):
        st = req.get("stage")
        rem = req.get("remaining_ms", 0)

        if st == "completed":
            survivors.append(req["id"])
        elif mode == "EXPLICIT":
            if st == "executing" and rem == 0:
                survivors.append(req["id"])
        elif mode == "GRACEFUL":
            if rem <= grace:
                survivors.append(req["id"])

    return sorted(survivors)


def derive_minimum_drain_timeout(config):
    queue_delay = config.get("queue_delay_ms", 0)
    durations = []

    for req in config.get("requests", []):
        st = req.get("stage")
        rem = req.get("remaining_ms", 0)

        if st == "completed":
            continue

        if st == "queued":
            durations.append(rem + queue_delay)
        else:
            durations.append(rem)

    if not durations:
        return 0
    return max(durations)
