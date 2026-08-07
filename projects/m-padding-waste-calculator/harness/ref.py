"""Oracle reference data and helper functions for grading harness."""

import numpy as np

DATASETS = [
    [128, 256, 512, 1024, 128, 64],
    [100, 100, 100, 100],
    [16, 32, 64, 128, 256, 512, 1024, 2048],
    [500, 1200, 300, 800, 1500, 400],
]

MAX_CAPACITIES = [1024, 500, 2048, 2000]


def ref_compute_padding_waste(seq_lens, max_len=None):
    lens = np.array(seq_lens, dtype=np.int64)
    if lens.size == 0:
        return 0.0
    target_len = int(max_len if max_len is not None else np.max(lens))
    total_padded = target_len * len(lens)
    if total_padded == 0:
        return 0.0
    total_real = int(np.sum(lens))
    return float((total_padded - total_real) / total_padded)


def ref_compute_flop_savings(seq_lens):
    lens = np.array(seq_lens, dtype=np.int64)
    if lens.size == 0:
        return 0.0
    max_l = int(np.max(lens))
    padded_flops = len(lens) * (max_l ** 2)
    if padded_flops == 0:
        return 0.0
    varlen_flops = int(np.sum(lens ** 2))
    return float((padded_flops - varlen_flops) / padded_flops)


def ref_pack_sequences_ffd(seq_lens, max_bin_capacity):
    indexed = sorted(enumerate(seq_lens), key=lambda x: x[1], reverse=True)
    bins = []
    for idx, length in indexed:
        if length > max_bin_capacity:
            raise ValueError("Sequence length exceeds max capacity")
        placed = False
        for b in bins:
            if sum(b["lengths"]) + length <= max_bin_capacity:
                b["indices"].append(idx)
                b["lengths"].append(length)
                placed = True
                break
        if not placed:
            bins.append({"indices": [idx], "lengths": [length]})
    return bins


def ref_compute_packing_efficiency(bins, max_bin_capacity):
    if not bins:
        return 0.0
    total_used = sum(sum(b["lengths"]) for b in bins)
    total_capacity = len(bins) * max_bin_capacity
    return float(total_used / total_capacity)


def ref_build_cu_seqlens(packed_bins):
    cu = [0]
    curr = 0
    for b in packed_bins:
        for l in b["lengths"]:
            curr += l
            cu.append(curr)
    return np.array(cu, dtype=np.int32)


def ref_build_sequence_metadata(packed_bins):
    all_lens = [l for b in packed_bins for l in b["lengths"]]
    if not all_lens:
        return {"total_tokens": 0, "max_seqlen": 0, "num_sequences": 0}
    return {
        "total_tokens": int(sum(all_lens)),
        "max_seqlen": int(max(all_lens)),
        "num_sequences": int(len(all_lens)),
    }
