import numpy as np


def train_qlora_step(qlora_layers, x, target, lr=0.01):
    keys = sorted(qlora_layers.keys())
    out = x
    activations = [out]

    for key in keys:
        out = qlora_layers[key].forward(out)
        activations.append(out)

    loss = np.mean((out - target) ** 2)
    grad = 2.0 * (out - target) / out.size

    for i in reversed(range(len(keys))):
        key = keys[i]
        layer = qlora_layers[key]
        act_in = activations[i]
        grad = layer.backward(act_in, grad)

        layer.lora_A -= lr * layer.grad_lora_A
        layer.lora_B -= lr * layer.grad_lora_B

    return loss


def run_qlora_training(qlora_layers, data_batches, lr=0.01):
    losses = []
    for x, target in data_batches:
        l = train_qlora_step(qlora_layers, x, target, lr=lr)
        losses.append(l)
    return losses


def verify_adapter_isolation(initial_base_weights, qlora_layers):
    for key, initial_weight in initial_base_weights.items():
        if key not in qlora_layers:
            continue
        layer = qlora_layers[key]
        reconstructed = layer.dequantize_base()

        dummy_x = np.ones((1, layer.in_features), dtype=np.float32)
        dummy_grad = np.ones((1, layer.out_features), dtype=np.float32)
        layer.backward(dummy_x, dummy_grad)

        if not hasattr(layer, "base_qweight") or layer.base_qweight is None:
            return False

        current_dequant = layer.dequantize_base()
        if not np.array_equal(reconstructed, current_dequant):
            return False

    return True
