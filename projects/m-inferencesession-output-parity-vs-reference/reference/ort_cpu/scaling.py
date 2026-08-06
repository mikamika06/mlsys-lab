import time
import numpy as np
import onnxruntime as ort


def measure_scaling(model_bytes, x_data, thread_counts):
    latencies = {}
    for t in thread_counts:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = t
        opts.inter_op_num_threads = 1
        session = ort.InferenceSession(model_bytes, opts, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name

        for _ in range(5):
            session.run(None, {input_name: x_data})

        start = time.perf_counter()
        iters = 50
        for _ in range(iters):
            session.run(None, {input_name: x_data})
        end = time.perf_counter()

        latencies[t] = float((end - start) / iters)
    return latencies
