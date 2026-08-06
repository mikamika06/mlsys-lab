import math


def min_workers_to_saturate(load_time, consumer_time):
    if consumer_time <= 0:
        return 1
    ratio = load_time / consumer_time
    return max(1, int(math.ceil(ratio)))
