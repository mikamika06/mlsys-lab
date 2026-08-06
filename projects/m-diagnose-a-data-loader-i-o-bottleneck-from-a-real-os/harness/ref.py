import random

IO_NAMES = ["read", "pread64", "write", "pwrite64", "openat", "close"]
SYNC_NAMES = ["futex", "epoll_wait", "poll", "select"]
OTHER_NAMES = ["mmap", "mprotect", "brk", "ioctl"]


def generate_osrt_dataset(seed):
    rng = random.Random(seed)
    num_rows = rng.randint(5, 12)
    rows = []

    names = rng.sample(IO_NAMES + SYNC_NAMES + OTHER_NAMES, min(num_rows, 12))
    for name in names:
        num_calls = rng.randint(10, 5000)
        avg_ms = rng.uniform(0.01, 5.0)
        total_ms = round(num_calls * avg_ms, 2)
        rows.append({
            "name": name,
            "num_calls": num_calls,
            "total_time_ms": total_ms,
            "avg_time_ms": avg_ms,
        })
    return rows


def generate_nvtx_events(seed):
    rng = random.Random(seed)
    events = []
    t = 1000
    names = ["data_loader", "fetch_batch", "transform", "cuda_memcpy", "decompress"]

    stack = []
    num_operations = rng.randint(10, 30)

    for _ in range(num_operations):
        t += rng.randint(10, 200)
        can_push = len(stack) < 4
        can_pop = len(stack) > 0

        if can_push and (not can_pop or rng.random() > 0.4):
            name = rng.choice(names)
            stack.append(name)
            events.append({"timestamp_ns": t, "event_type": "push", "name": name})
        elif can_pop:
            name = stack.pop()
            events.append({"timestamp_ns": t, "event_type": "pop", "name": name})

    while stack:
        t += rng.randint(10, 200)
        name = stack.pop()
        events.append({"timestamp_ns": t, "event_type": "pop", "name": name})

    return events
