def insert_per_channel_qdq(tensor_info):
    name = tensor_info["name"]
    shape = tensor_info["shape"]
    axis = tensor_info["axis"]
    scale_shape = (shape[axis],) if axis < len(shape) else (1,)
    return {
        "q_node": f"Quantize_{name}",
        "dq_node": f"Dequantize_{name}",
        "scale_shape": scale_shape,
        "axis": axis
    }
