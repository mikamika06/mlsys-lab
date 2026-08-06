import random

def generate_fixtures():
    random.seed(42)
    traces = []
    for _ in range(5):
        e_ops = random.randint(10, 50)
        e_size = random.randint(500, 2000)
        c_ops = random.randint(5, e_ops)
        c_size = random.randint(100, e_size)
        traces.append(({"ops": e_ops, "size": e_size}, {"ops": c_ops, "size": c_size}))
    sequences = [
        [1, 2, 4, 1, 2, 4, 8, 1, 2, 4],
        [8, 8, 8, 1, 1, 8, 8],
        [1, 2, 3, 4, 5, 1, 2]
    ]
    return traces, sequences
