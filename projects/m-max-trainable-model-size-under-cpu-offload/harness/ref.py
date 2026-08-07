import numpy as np

SCENARIOS = [
    {
        "gpu_mem_bytes": 16 * 1024**3,
        "cpu_mem_bytes": 64 * 1024**3,
        "seq_len": 2048,
        "batch_size": 2,
        "hidden_dim": 4096,
        "num_gpus": 8
    },
    {
        "gpu_mem_bytes": 32 * 1024**3,
        "cpu_mem_bytes": 256 * 1024**3,
        "seq_len": 4096,
        "batch_size": 4,
        "hidden_dim": 8192,
        "num_gpus": 16
    },
    {
        "gpu_mem_bytes": 80 * 1024**3,
        "cpu_mem_bytes": 512 * 1024**3,
        "seq_len": 8192,
        "batch_size": 1,
        "hidden_dim": 4096,
        "num_gpus": 32
    }
]

def max_trainable_model_size(gpu_mem_bytes, cpu_mem_bytes, seq_len, batch_size, hidden_dim, num_gpus):
    params_per_layer = 12 * hidden_dim * hidden_dim
    act_per_layer = batch_size * seq_len * hidden_dim * 34
    max_l = 0
    for l in range(1, 10000):
        total_params = l * params_per_layer
        gpu_param_mem = (2 * total_params) / num_gpus
        gpu_grad_mem = (2 * total_params) / num_gpus
        gpu_working_mem = 2 * params_per_layer
        gpu_act_mem = l * act_per_layer
        total_gpu = gpu_param_mem + gpu_grad_mem + gpu_working_mem + gpu_act_mem
        cpu_opt_mem = (16 * total_params) / num_gpus
        if total_gpu <= gpu_mem_bytes and cpu_opt_mem <= cpu_mem_bytes:
            max_l = l
        else:
            break
    return max_l * params_per_layer

def int8_block_quantize(tensor, block_size=64):
    flat = np.asarray(tensor, dtype=np.float32).flatten()
    orig_len = len(flat)
    remainder = orig_len % block_size
    pad_len = (block_size - remainder) if remainder != 0 else 0
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len), mode='constant', constant_values=0)
    blocks = flat.reshape(-1, block_size)
    max_vals = np.max(np.abs(blocks), axis=1, keepdims=True)
    max_vals = np.maximum(max_vals, 1e-8)
    scales = (max_vals / 127.0).astype(np.float32)
    q_blocks = np.clip(np.round(blocks / scales), -128, 127).astype(np.int8)
    return q_blocks, scales.squeeze(-1)

def int8_block_dequantize(q_blocks, scales, original_shape):
    scales_col = scales.reshape(-1, 1).astype(np.float32)
    dequant = q_blocks.astype(np.float32) * scales_col
    flat = dequant.flatten()
    total_elements = int(np.prod(original_shape))
    return flat[:total_elements].reshape(original_shape)

def zero_plus_comm_volume(num_params, bytes_per_param, world_size, num_nodes, enable_hpz=True, enable_qgz=True):
    base_volume = 2.0 * num_params * bytes_per_param * (world_size - 1.0) / world_size
    hpz_factor = 0.5 if (enable_hpz and num_nodes > 1) else 1.0
    qgz_factor = 0.5 if enable_qgz else 1.0
    return base_volume * hpz_factor * qgz_factor
