def detect_wrong_dimension(tensor_shape, quantized_dim):
    if quantized_dim < 0 or quantized_dim >= len(tensor_shape):
        return True
    if tensor_shape[quantized_dim] == 1 and len(tensor_shape) > 1:
        return True
    return False
