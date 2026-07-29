import time


def measure(fn, warmup=0, reps=3, timer=time.perf_counter, sync=None):
    t0 = timer()
    for _ in range(reps):
        fn()
    return {"mean": (timer() - t0) / reps}
