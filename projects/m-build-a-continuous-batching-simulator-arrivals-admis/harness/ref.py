import random

class Request:
    def __init__(self, req_id, arrival, prompt_len, decode_len):
        self.id = req_id
        self.arrival = arrival
        self.prompt_len = prompt_len
        self.decode_len = decode_len

def generate_trace(n, seed=42):
    random.seed(seed)
    trace = []
    t = 0
    for i in range(n):
        arrival = t
        t += random.randint(0, 5)
        prompt = random.randint(10, 100)
        if random.random() < 0.15:
            decode = random.randint(150, 400)
        else:
            decode = random.randint(5, 25)
        trace.append(Request(i, arrival, prompt, decode))
    return trace

def simulate_static(requests, max_batch_size):
    pending = list(requests)
    running = []
    tick = 0
    log = []
    while pending or running:
        if not running:
            available = [r for r in pending if r.arrival <= tick]
            if not available and pending:
                tick = pending[0].arrival
                available = [r for r in pending if r.arrival <= tick]
            for r in available[:max_batch_size]:
                pending.remove(r)
                running.append(r.decode_len)
        log.append(len(running))
        running = [rem - 1 for rem in running if rem > 1]
        tick += 1
    return tick, log

def simulate_continuous(requests, max_batch_size):
    pending = list(requests)
    running = []
    tick = 0
    log = []
    while pending or running:
        available = [r for r in pending if r.arrival <= tick]
        if not available and not running and pending:
            tick = pending[0].arrival
            available = [r for r in pending if r.arrival <= tick]
        for r in available[:max_batch_size - len(running)]:
            pending.remove(r)
            running.append(r.decode_len)
        log.append(len(running))
        running = [rem - 1 for rem in running if rem > 1]
        tick += 1
    return tick, log

def compare_throughput(requests, max_batch_size):
    static_ticks, _ = simulate_static(requests, max_batch_size)
    continuous_ticks, _ = simulate_continuous(requests, max_batch_size)
    if continuous_ticks == 0:
        return 0.0
    return float(static_ticks) / float(continuous_ticks)

def occupancy_histogram(log, max_batch_size):
    hist = [0] * max_batch_size
    for sz in log:
        if 1 <= sz <= max_batch_size:
            hist[sz - 1] += 1
    return hist
