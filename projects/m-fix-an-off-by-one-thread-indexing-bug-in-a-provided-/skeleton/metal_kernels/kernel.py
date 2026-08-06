def run_indexing_kernel(size):
    raise NotImplementedError

def run_sum_reduction_kernel(arr, math_mode="safe"):
    raise NotImplementedError

def run_dequant_kernel(packed, scales, biases):
    raise NotImplementedError
