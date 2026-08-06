def group_tensors_by_device_dtype(params):
    """
    Group tensors by (device, dtype).

    params: list of dicts with keys 'id', 'dtype', 'device', 'shape'
    returns: dict mapping (device, dtype) tuple to list of param dicts
    """
    raise NotImplementedError


def estimate_kernel_counts(params, num_steps=1):
    """
    Estimate CUDA kernel launches for 'loop', 'foreach', and 'fused' Adam over num_steps.

    - loop: 4 kernels per parameter per step (grad scaling, m update, v update, param update)
    - foreach: 4 kernels per tensor group per step
    - fused: 1 kernel per tensor group per step

    returns: dict with keys 'loop', 'foreach', 'fused' mapping to total kernel count
    """
    raise NotImplementedError
