import torch

def inspect_autocast(model, x, device_type, autocast_dtype):
    """
    Runs a forward pass of `model` on `x` using torch.autocast.
    Returns a dictionary with:
    - "output_dtype": the dtype of the returned tensor
    - "activation_dtypes": list of dtypes for each leaf module's output, in execution order
    - "weight_dtypes": list of dtypes for each leaf module's .weight parameter, in order of model.named_modules()
    """
    raise NotImplementedError

def synthesize_overflow():
    """
    Returns (a, b) as 1D torch.Tensor (float32) such that their element-wise product sum
    overflows in float16 but is safely representable in bfloat16.
    """
    raise NotImplementedError
