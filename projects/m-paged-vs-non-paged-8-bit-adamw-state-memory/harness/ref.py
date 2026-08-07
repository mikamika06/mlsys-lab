import math
import numpy as np

CODEBOOK_4BIT = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
PAGE_TABLE_OVERHEAD_BYTES = 1048576


def compute_adamw_state_bytes(num_params: int, block_size: int = 256, paged: bool = False, max_layer_params: int = 0) -> int:
    if num_params <= 0:
        return 0
    blocks = math.ceil(num_params / block_size)
    non_paged_bytes = num_params * 2 + blocks * 8
    if not paged:
        return non_paged_bytes
    target_layer_params = max_layer_params if max_layer_params > 0 else num_params
    layer_blocks = math.ceil(target_layer_params / block_size)
    layer_bytes = target_layer_params * 2 + layer_blocks * 8
    paged_vram = layer_bytes + PAGE_TABLE_OVERHEAD_BYTES
    return min(non_paged_bytes, paged_vram)


def qlora_peak_memory_plan(config: dict) -> dict:
    base_params = config["base_params"]
    lora_params = config["lora_params"]
    max_layer_base = config.get("max_layer_base_params", base_params)
    max_layer_lora = config.get("max_layer_lora_params", lora_params)
    seq_len = config["seq_len"]
    batch_size = config["batch_size"]
    hidden_dim = config["hidden_dim"]
    num_layers = config["num_layers"]
    paged_adamw = config.get("paged_adamw", False)
    grad_ckpt = config.get("gradient_checkpointing", False)
    vram_gb = config["vram_gb"]

    base_weight_bytes = math.ceil(base_params * 0.5) + math.ceil(base_params / 64) * 2
    lora_weight_bytes = lora_params * 2
    gradient_bytes = lora_params * 2
    optimizer_bytes = compute_adamw_state_bytes(lora_params, block_size=256, paged=paged_adamw, max_layer_params=max_layer_lora)

    per_layer_act = batch_size * seq_len * hidden_dim * 20
    if grad_ckpt:
        activation_bytes = (num_layers * batch_size * seq_len * hidden_dim * 2) + per_layer_act
    else:
        activation_bytes = num_layers * per_layer_act

    workspace_bytes = max_layer_base * 2

    peak_vram_bytes = (
        base_weight_bytes + lora_weight_bytes + gradient_bytes +
        optimizer_bytes + activation_bytes + workspace_bytes
    )
    limit_bytes = int(vram_gb * (1024 ** 3))
    fits_in_vram = bool(peak_vram_bytes <= limit_bytes)

    return {
        "base_weight_bytes": base_weight_bytes,
        "lora_weight_bytes": lora_weight_bytes,
        "gradient_bytes": gradient_bytes,
        "optimizer_bytes": optimizer_bytes,
        "activation_bytes": activation_bytes,
        "workspace_bytes": workspace_bytes,
        "peak_vram_bytes": peak_vram_bytes,
        "fits_in_vram": fits_in_vram,
    }


def dequantize_4bit(qweights: np.ndarray, scales: np.ndarray, block_size: int = 64) -> np.ndarray:
    m, n = qweights.shape
    flat_q = qweights.reshape(-1)
    flat_scales = scales.reshape(-1)
    num_blocks = len(flat_q) // block_size
    dequant = CODEBOOK_4BIT[flat_q].astype(np.float32)
    dequant = dequant.reshape(num_blocks, block_size)
    scaled = dequant * flat_scales[:, np.newaxis]
    return scaled.reshape(m, n)


def merge_lora_into_base(qweights: np.ndarray, scales: np.ndarray, lora_A: np.ndarray, lora_B: np.ndarray, alpha: float, block_size: int = 64) -> np.ndarray:
    w_base = dequantize_4bit(qweights, scales, block_size)
    r = lora_A.shape[0]
    scaling = alpha / float(r)
    delta = (lora_B @ lora_A) * scaling
    return w_base + delta


def quantize_to_4bit(weights: np.ndarray, block_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    m, n = weights.shape
    flat_w = weights.reshape(-1).astype(np.float32)
    total_elements = len(flat_w)
    num_blocks = total_elements // block_size
    blocks = flat_w.reshape(num_blocks, block_size)
    max_vals = np.max(np.abs(blocks), axis=1)
    scales = np.where(max_vals == 0, 1.0, max_vals).astype(np.float32)
    norm_blocks = blocks / scales[:, np.newaxis]
    diffs = np.abs(norm_blocks[:, :, np.newaxis] - CODEBOOK_4BIT[np.newaxis, np.newaxis, :])
    q_indices = np.argmin(diffs, axis=2).astype(np.uint8)
    qweights = q_indices.reshape(m, n)
    return qweights, scales


def generate_configs():
    return [
        {
            "base_params": 7_000_000_000,
            "lora_params": 20_000_000,
            "max_layer_base_params": 220_000_000,
            "max_layer_lora_params": 650_000,
            "seq_len": 2048,
            "batch_size": 2,
            "hidden_dim": 4096,
            "num_layers": 32,
            "paged_adamw": False,
            "gradient_checkpointing": True,
            "vram_gb": 16.0,
        },
        {
            "base_params": 7_000_000_000,
            "lora_params": 20_000_000,
            "max_layer_base_params": 220_000_000,
            "max_layer_lora_params": 650_000,
            "seq_len": 2048,
            "batch_size": 2,
            "hidden_dim": 4096,
            "num_layers": 32,
            "paged_adamw": True,
            "gradient_checkpointing": True,
            "vram_gb": 16.0,
        },
        {
            "base_params": 13_000_000_000,
            "lora_params": 40_000_000,
            "max_layer_base_params": 320_000_000,
            "max_layer_lora_params": 1_000_000,
            "seq_len": 4096,
            "batch_size": 1,
            "hidden_dim": 5120,
            "num_layers": 40,
            "paged_adamw": True,
            "gradient_checkpointing": True,
            "vram_gb": 24.0,
        },
    ]


def generate_quant_fixture(seed: int = 42):
    rng = np.random.default_rng(seed)
    m, n = 128, 256
    block_size = 64
    qweights = rng.integers(0, 16, size=(m, n), dtype=np.uint8)
    num_blocks = (m * n) // block_size
    scales = rng.uniform(0.1, 2.0, size=num_blocks).astype(np.float32)
    r = 16
    lora_A = rng.normal(0, 0.02, size=(r, n)).astype(np.float32)
    lora_B = rng.normal(0, 0.02, size=(m, r)).astype(np.float32)
    alpha = 32.0
    return qweights, scales, lora_A, lora_B, alpha, block_size
