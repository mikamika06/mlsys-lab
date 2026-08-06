def compute_nhwc_strides(shape):
    N, C, H, W = shape
    return (H * W * C, 1, W * C, C)


def steady_state_batch_time(cpu_prep_ms, transfer_ms, gpu_compute_ms, pin_memory, non_blocking):
    is_async = pin_memory and non_blocking
    if is_async:
        return max(cpu_prep_ms, transfer_ms + gpu_compute_ms)
    else:
        return max(cpu_prep_ms + transfer_ms, gpu_compute_ms)


SHAPES = [
    (1, 3, 224, 224),
    (32, 64, 56, 56),
    (16, 128, 28, 28)
]

IS_CL_CASES = [
    ((1, 3, 224, 224), compute_nhwc_strides((1, 3, 224, 224)), True),
    ((1, 3, 224, 224), (3 * 224 * 224, 224 * 224, 224, 1), False),
    ((2, 5, 10, 10), compute_nhwc_strides((2, 5, 10, 10)), True),
    ((2, 5, 10, 10), (1000, 1, 50, 5), False),
]

PIPELINE_CASES = [
    (10, 10, 5, False, False),
    (10, 10, 5, False, True),
    (10, 10, 5, True, False),
    (10, 10, 5, True, True),
    (5, 15, 20, False, True),
    (5, 15, 20, True, True),
]
