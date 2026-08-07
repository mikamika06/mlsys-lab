import numpy as np


def count_recompiles(shapes, bucket_size=64):
    seen = set()
    recompiles = 0
    for s in shapes:
        b = ((s + bucket_size - 1) // bucket_size) * bucket_size
        if b not in seen:
            seen.add(b)
            recompiles += 1
    return recompiles


def bucket_shape(seq_len, bucket_sizes):
    for b in sorted(bucket_sizes):
        if seq_len <= b:
            return b
    return bucket_sizes[-1]


def capture_decode_step(func, sample_inputs):
    class MockGraph:
        def __init__(self, inputs):
            self.inputs = inputs
            self.captured = True

        def replay(self):
            return func(*self.inputs)

    return MockGraph(sample_inputs)


SHAPES_FIXTURE = [12, 15, 31, 64, 65, 120, 128, 200]
BUCKETS = [64, 128, 256, 512]
