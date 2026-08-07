import numpy as np


class QLoRALinear:
    """Simulated 4-bit Block-Quantized Linear layer with LoRA adapter."""

    def __init__(self, in_features, out_features, r=4, alpha=8.0, block_size=16):
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.block_size = block_size

        self.lora_A = np.zeros((in_features, r), dtype=np.float32)
        self.lora_B = np.zeros((r, out_features), dtype=np.float32)
        self.grad_lora_A = np.zeros_like(self.lora_A)
        self.grad_lora_B = np.zeros_like(self.lora_B)

        self.base_qweight = None
        self.block_scales = None
        self.shape = (in_features, out_features)

    def quantize_base(self, weight):
        self.shape = weight.shape
        flat = weight.astype(np.float32).flatten()
        n = flat.size
        pad_len = (self.block_size - (n % self.block_size)) % self.block_size
        if pad_len > 0:
            flat_padded = np.pad(flat, (0, pad_len), mode="constant")
        else:
            flat_padded = flat

        blocks = flat_padded.reshape(-1, self.block_size)
        max_abs = np.max(np.abs(blocks), axis=1, keepdims=True)
        scales = np.where(max_abs == 0, 1.0, max_abs / 7.0)

        q_blocks = np.round(blocks / scales)
        q_blocks = np.clip(q_blocks, -8, 7).astype(np.int8)

        self.base_qweight = q_blocks.flatten()[:n]
        self.block_scales = scales.squeeze(-1).astype(np.float32)

    def dequantize_base(self):
        flat_q = self.base_qweight
        n = flat_q.size
        pad_len = (self.block_size - (n % self.block_size)) % self.block_size
        if pad_len > 0:
            flat_padded = np.pad(flat_q, (0, pad_len), mode="constant")
        else:
            flat_padded = flat_q

        blocks = flat_padded.reshape(-1, self.block_size)
        scales = self.block_scales[:, None]
        dequant = (blocks.astype(np.float32) * scales).flatten()[:n]
        return dequant.reshape(self.shape)

    def forward(self, x):
        w_dequant = self.dequantize_base()
        base_out = np.matmul(x, w_dequant)
        lora_out = np.matmul(np.matmul(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out

    def backward(self, x, grad_output):
        self.grad_lora_A = np.matmul(x.T, np.matmul(grad_output, self.lora_B.T)) * self.scaling
        self.grad_lora_B = np.matmul(np.matmul(x, self.lora_A).T, grad_output) * self.scaling
        w_dequant = self.dequantize_base()
        grad_x = np.matmul(grad_output, w_dequant.T)
        return grad_x


def apply_qlora(model_dict, target_keys, r=4, alpha=8.0, block_size=16):
    qlora_layers = {}
    for key, weight in model_dict.items():
        if key in target_keys:
            in_f, out_f = weight.shape
            layer = QLoRALinear(in_f, out_f, r=r, alpha=alpha, block_size=block_size)
            rng = np.random.RandomState(abs(hash(key)) % (2**31))
            layer.lora_A = (rng.randn(in_f, r) * 0.01).astype(np.float32)
            layer.lora_B = np.zeros((r, out_f), dtype=np.float32)
            layer.quantize_base(weight)
            qlora_layers[key] = layer
    return qlora_layers
