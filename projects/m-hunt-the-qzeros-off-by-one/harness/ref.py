import numpy as np


def generate_cases():
    cases = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        qzeros = rng.integers(0, 15, size=(8, 8), dtype=np.int32)
        group_size = 32
        out = qzeros.copy()
        out = np.clip(out + 1, 0, None)
        cases.append({"qzeros": qzeros, "group_size": group_size, "expected": out})
    return cases


CONFIGS = generate_cases()


def fix_qzeros(qzeros, group_size):
    out = qzeros.copy()
    out = np.clip(out + 1, 0, None)
    return out


def apply_gidx(weight, g_idx):
    order = np.argsort(g_idx)
    return weight[:, order]


def invert_gidx(permuted_weight, g_idx):
    order = np.argsort(g_idx)
    inv_order = np.argsort(order)
    return permuted_weight[:, inv_order]


def compute_packing_sizes(num_weights, bits):
    total_bits = num_weights * bits
    bytes_unaligned = int(np.ceil(total_bits / 8.0))
    bytes_aligned = int(np.ceil(bytes_unaligned / 128.0) * 128)
    return {"unaligned_bytes": bytes_unaligned, "aligned_bytes": bytes_aligned}
