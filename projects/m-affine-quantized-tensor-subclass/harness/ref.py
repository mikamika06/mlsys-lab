import numpy as np


def generate_weight():
    np.random.seed(42)
    return np.random.randn(128, 256).astype(np.float32)


def generate_act():
    np.random.seed(43)
    return np.random.randn(16, 256).astype(np.float32)


class RefAffineQuantizedTensor:
    def __init__(self, int_data, scale, zero_point, shape, group_size, asymmetric):
        self.int_data = int_data
        self.scale = scale
        self.zero_point = zero_point
        self.shape = shape
        self.group_size = group_size
        self.asymmetric = asymmetric

    def dequantize(self):
        if self.asymmetric:
            deq = (self.int_data.astype(np.float32) - self.zero_point) * self.scale
        else:
            deq = self.int_data.astype(np.float32) * self.scale
        return deq.reshape(self.shape)

    def __matmul__(self, other):
        return self.dequantize() @ other


def quantize_affine(weight, group_size, asymmetric):
    shape = weight.shape
    w = weight.reshape(-1, group_size)
    if asymmetric:
        w_min = w.min(axis=1, keepdims=True)
        w_max = w.max(axis=1, keepdims=True)
        scale = (w_max - w_min) / 15.0
        scale[scale == 0] = 1.0
        zero_point = np.round(-w_min / scale)
        int_data = np.clip(np.round(w / scale + zero_point), 0, 15).astype(np.uint8)
        return RefAffineQuantizedTensor(int_data, scale, zero_point, shape, group_size, asymmetric)
    else:
        w_max = np.abs(w).max(axis=1, keepdims=True)
        scale = w_max / 7.0
        scale[scale == 0] = 1.0
        int_data = np.clip(np.round(w / scale), -7, 7).astype(np.int8)
        return RefAffineQuantizedTensor(int_data, scale, None, shape, group_size, asymmetric)


def map_target_to_config(target):
    if target == "edge_device":
        return {"method": "torchao_int4", "group_size": 32, "asymmetric": True}
    elif target == "fine_tuning":
        return {"method": "bnb_nf4", "group_size": 64, "asymmetric": False}
    elif target == "server_inference":
        return {"method": "gptq_w4a16", "group_size": 128, "asymmetric": True}
    raise ValueError(f"Unknown target {target}")


def get_rel_err(weight, target):
    config = map_target_to_config(target)
    q_tensor = quantize_affine(weight, config["group_size"], config["asymmetric"])
    deq = q_tensor.dequantize()
    return float(np.linalg.norm(deq - weight) / np.linalg.norm(weight))
