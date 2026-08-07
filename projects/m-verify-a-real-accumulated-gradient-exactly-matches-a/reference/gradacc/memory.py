import tracemalloc
import gc
from gradacc.accumulate import accumulate


def peak_memory_sweep(micro_batch_generator, W, b, steps_list):
    peaks = []
    for steps in steps_list:
        gc.collect()
        batches = micro_batch_generator(steps)
        tracemalloc.start()
        accumulate(batches, W, b, steps)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)
    return peaks
