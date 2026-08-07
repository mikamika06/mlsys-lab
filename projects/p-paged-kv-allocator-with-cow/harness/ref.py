import random

def get_random_trace(steps: int, seed: int = 42) -> list:
    random.seed(seed)
    trace = []
    active = []
    seq_id = 1

    for _ in range(steps):
        op = random.random()
        if not active or op < 0.1:
            trace.append(("alloc", seq_id))
            active.append(seq_id)
            seq_id += 1
        elif op < 0.5:
            target = random.choice(active)
            trace.append(("append", target))
        elif op < 0.7:
            target = random.choice(active)
            trace.append(("fork", target, seq_id))
            active.append(seq_id)
            seq_id += 1
        else:
            target = random.choice(active)
            active.remove(target)
            trace.append(("free", target))

    for s in active:
        trace.append(("free", s))

    return trace

def get_beam_search_trace() -> list:
    trace = []
    trace.append(("alloc", 0))
    for _ in range(100):
        trace.append(("append", 0))

    active = []
    seq_id = 1
    for _ in range(4):
        trace.append(("fork", 0, seq_id))
        active.append(seq_id)
        seq_id += 1
    trace.append(("free", 0))

    for _ in range(20):
        for s in active:
            trace.append(("append", s))

        trace.append(("free", active[0]))
        trace.append(("free", active[1]))

        s1 = active[2]
        s2 = active[3]

        trace.append(("fork", s1, seq_id))
        new1 = seq_id
        seq_id += 1

        trace.append(("fork", s2, seq_id))
        new2 = seq_id
        seq_id += 1

        active = [s1, s2, new1, new2]

    for s in active:
        trace.append(("free", s))

    return trace
