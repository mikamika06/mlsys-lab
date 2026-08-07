import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    res = {"pack_ok": 0.0, "leakage_detected": 0.0, "zero_leakage_on_block_diag": 0.0}

    try:
        from seqpack.pack import pack_sequences, measure_attention_leakage
    except Exception:
        return res

    data = ref.get_synthetic_data(num_seqs=4, seed=42)
    max_len = 40
    try:
        packed = pack_sequences(data, max_seq_len=max_len)
    except Exception:
        return res

    if not packed or not isinstance(packed, list):
        return res

    row0 = packed[0]
    required_keys = {"input_ids", "labels", "label_mask", "seq_ids", "position_ids"}
    if not required_keys.issubset(row0.keys()):
        return res

    if len(row0["input_ids"]) != max_len or len(row0["seq_ids"]) != max_len:
        return res

    seq_ids = row0["seq_ids"]
    pos_ids = row0["position_ids"]
    unique_seqs = set(seq_ids[seq_ids >= 0])

    if len(unique_seqs) >= 2:
        seq1_idx = np.where(seq_ids == 1)[0]
        if len(seq1_idx) > 0 and pos_ids[seq1_idx[0]] == 0:
            res["pack_ok"] = 1.0
    else:
        res["pack_ok"] = 1.0

    L = max_len
    naive_attn = np.tril(np.ones((L, L))) / np.arange(1, L + 1)[:, None]

    try:
        leak = measure_attention_leakage(naive_attn, seq_ids)
        if leak > 0.15:
            res["leakage_detected"] = 1.0
    except Exception:
        pass

    block_mask = ref.create_block_diagonal_mask(seq_ids)
    clean_attn = naive_attn * block_mask
    row_sums = np.sum(clean_attn, axis=-1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    clean_attn = clean_attn / row_sums

    try:
        zero_leak = measure_attention_leakage(clean_attn, seq_ids)
        if abs(zero_leak) < 1e-6:
            res["zero_leakage_on_block_diag"] = 1.0
    except Exception:
        pass

    return res
