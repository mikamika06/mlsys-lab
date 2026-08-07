import time

def measure_latency(fn, args, warmup=10, reps=50):
    for _ in range(warmup):
        fn()
    start = time.time_ns()
    for _ in range(reps):
        fn()
    end = time.time_ns()
    return (end - start) / float(reps)
