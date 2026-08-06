import math

def layer_bytes(shape, method):
    c_out = shape[0]
    n = math.prod(shape)
    w = n // c_out

    if method == "float16":
        return n * 2
    elif method == "int8_channel":
        return n + c_out * 2
    elif method == "lut4_channel_fp16":
        return c_out * math.ceil(w / 2) + c_out * 32
    elif method == "lut4_joint_int8_channel":
        return c_out * math.ceil(w / 2) + c_out * 18
    elif method == "lut8_channel_fp16":
        return n + c_out * 512
    elif method == "lut8_joint_int8_channel":
        return n + c_out * 258
    else:
        raise ValueError(f"Unknown method {method}")

def optimize_model(shapes, allowed_methods):
    plan = []
    tot = 0
    for shape in shapes:
        best_sz = float('inf')
        best_m = None
        for m in allowed_methods:
            sz = layer_bytes(shape, m)
            if sz < best_sz:
                best_sz = sz
                best_m = m
        plan.append(best_m)
        tot += best_sz
    return plan, tot
