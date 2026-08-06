import random


def get_fixtures():
    random.seed(42)
    seqs = [random.randint(10, 1000) for _ in range(100)]
    traces = []
    base_prompt = [random.randint(0, 1000) for _ in range(50)]
    for _ in range(20):
        trace = base_prompt[:]
        trace.extend([random.randint(0, 1000) for _ in range(random.randint(10, 50))])
        traces.append(trace)
    return seqs, traces


def optimal_block_size(sequence_lengths, max_b):
    best_b = 1
    min_frag = float('inf')
    for b in range(1, max_b + 1):
        frag = sum(((l + b - 1) // b) * b - l for l in sequence_lengths)
        if frag < min_frag:
            min_frag = frag
            best_b = b
    return best_b


def measure_hit_rate(traces, num_blocks, block_size):
    alloc = list(range(num_blocks))
    cache = {}
    hits = 0
    misses = 0
    for tokens in traces:
        parent = -1
        num_full = len(tokens) // block_size
        for i in range(num_full):
            chunk = tuple(tokens[i * block_size : (i + 1) * block_size])
            key = (parent, chunk)
            if key in cache:
                hits += 1
                parent = cache[key]
            else:
                if not alloc:
                    break
                phys = alloc.pop()
                cache[key] = phys
                parent = phys
                misses += 1
    if hits + misses == 0:
        return 0.0
    return hits / (hits + misses)
