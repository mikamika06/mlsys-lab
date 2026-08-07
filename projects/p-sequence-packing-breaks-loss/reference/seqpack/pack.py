import numpy as np


def pack_sequences(sequences, max_seq_len, pad_token_id=0):
    """Packs multiple sequence dicts into fixed-length rows."""
    packed_rows = []

    curr_input_ids = []
    curr_labels = []
    curr_label_mask = []
    curr_seq_ids = []
    curr_position_ids = []

    seq_idx_in_row = 0
    curr_len = 0

    for seq in sequences:
        inp = np.asarray(seq["input_ids"], dtype=np.int64)
        lbl = np.asarray(seq["labels"], dtype=np.int64)
        msk = np.asarray(seq["label_mask"], dtype=np.float32)
        n = len(inp)

        if curr_len + n > max_seq_len and curr_len > 0:
            pad_len = max_seq_len - curr_len
            p_ids = np.pad(np.concatenate(curr_input_ids), (0, pad_len), constant_values=pad_token_id)
            p_lbl = np.pad(np.concatenate(curr_labels), (0, pad_len), constant_values=-100)
            p_msk = np.pad(np.concatenate(curr_label_mask), (0, pad_len), constant_values=0.0)
            p_seq = np.pad(np.concatenate(curr_seq_ids), (0, pad_len), constant_values=-1)
            p_pos = np.pad(np.concatenate(curr_position_ids), (0, pad_len), constant_values=0)

            packed_rows.append({
                "input_ids": p_ids,
                "labels": p_lbl,
                "label_mask": p_msk,
                "seq_ids": p_seq,
                "position_ids": p_pos,
            })

            curr_input_ids = []
            curr_labels = []
            curr_label_mask = []
            curr_seq_ids = []
            curr_position_ids = []
            seq_idx_in_row = 0
            curr_len = 0

        actual_n = min(n, max_seq_len)
        curr_input_ids.append(inp[:actual_n])
        curr_labels.append(lbl[:actual_n])
        curr_label_mask.append(msk[:actual_n])
        curr_seq_ids.append(np.full(actual_n, seq_idx_in_row, dtype=np.int64))
        curr_position_ids.append(np.arange(actual_n, dtype=np.int64))

        seq_idx_in_row += 1
        curr_len += actual_n

    if curr_len > 0:
        pad_len = max_seq_len - curr_len
        p_ids = np.pad(np.concatenate(curr_input_ids), (0, pad_len), constant_values=pad_token_id)
        p_lbl = np.pad(np.concatenate(curr_labels), (0, pad_len), constant_values=-100)
        p_msk = np.pad(np.concatenate(curr_label_mask), (0, pad_len), constant_values=0.0)
        p_seq = np.pad(np.concatenate(curr_seq_ids), (0, pad_len), constant_values=-1)
        p_pos = np.pad(np.concatenate(curr_position_ids), (0, pad_len), constant_values=0)

        packed_rows.append({
            "input_ids": p_ids,
            "labels": p_lbl,
            "label_mask": p_msk,
            "seq_ids": p_seq,
            "position_ids": p_pos,
        })

    return packed_rows


def measure_attention_leakage(attn_weights, seq_ids):
    """Measures attention weight leaking across distinct sequence IDs."""
    attn = np.asarray(attn_weights, dtype=np.float64)
    s_ids = np.asarray(seq_ids, dtype=np.int64)

    if attn.ndim == 2:
        valid_mask = (s_ids[:, None] >= 0) & (s_ids[None, :] >= 0)
        diff_seq_mask = valid_mask & (s_ids[:, None] != s_ids[None, :])
        total_mass = np.sum(attn * valid_mask)
        if total_mass == 0:
            return 0.0
        leaked_mass = np.sum(attn * diff_seq_mask)
        return float(leaked_mass / total_mass)

    elif attn.ndim == 3:
        valid_mask = (s_ids[:, None] >= 0) & (s_ids[None, :] >= 0)
        diff_seq_mask = valid_mask & (s_ids[:, None] != s_ids[None, :])
        total_mass = np.sum(attn * valid_mask[None, :, :])
        if total_mass == 0:
            return 0.0
        leaked_mass = np.sum(attn * diff_seq_mask[None, :, :])
        return float(leaked_mass / total_mass)

    elif attn.ndim == 4:
        B, H, L, _ = attn.shape
        if s_ids.ndim == 1:
            s_ids = np.tile(s_ids, (B, 1))
        leaked_sum = 0.0
        total_sum = 0.0
        for b in range(B):
            v_mask = (s_ids[b, :, None] >= 0) & (s_ids[b, None, :] >= 0)
            d_mask = v_mask & (s_ids[b, :, None] != s_ids[b, None, :])
            tot = np.sum(attn[b] * v_mask[None, :, :])
            leak = np.sum(attn[b] * d_mask[None, :, :])
            total_sum += tot
            leaked_sum += leak
        if total_sum == 0:
            return 0.0
        return float(leaked_sum / total_sum)

    return 0.0
