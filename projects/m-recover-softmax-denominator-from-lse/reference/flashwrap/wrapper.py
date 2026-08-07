def flash_wrapper(q, k, v, scale=None, qkvpacked=False):
    if qkvpacked:
        if q is not None and len(q.shape) == 4:
            return {"mode": "packed", "shape": q.shape, "scale": scale}
        raise ValueError("Invalid packed tensor shape")
    if q is not None and k is not None and v is not None:
        return {"mode": "unpacked", "q_shape": q.shape, "k_shape": k.shape, "v_shape": v.shape, "scale": scale}
    raise ValueError("Missing required tensors")
