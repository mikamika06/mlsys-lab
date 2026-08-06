import time


def measure_kernel(func, warmup=10, reps=50):
    for _ in range(warmup):
        func()

    start_time = time.perf_counter()
    for _ in range(reps):
        func()
    end_time = time.perf_counter()

    return (end_time - start_time) / reps * 1000.0
