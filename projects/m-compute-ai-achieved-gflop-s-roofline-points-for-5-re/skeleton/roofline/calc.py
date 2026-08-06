def compute_roofline_points(kernels, peak_flops, peak_bw):
    raise NotImplementedError


def classify_kernels(metrics):
    raise NotImplementedError


def compare_attention(std_metrics, flash_metrics):
    raise NotImplementedError


def find_limiter(kernel_metric, peak_flops, peak_bw):
    raise NotImplementedError
