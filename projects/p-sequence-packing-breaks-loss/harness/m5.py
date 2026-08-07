import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    res = {
        "padding_overhead_reduced": 0.0,
        "loss_divergence_zero": 0.0,
        "speedup_achieved": 0.0,
    }

    try:
        from seqpack.pack import pack_sequences
        from seqpack.attention import compute_packed_attention
        from seqpack.loss import compute_packed_loss
    except Exception:
        return res

    rng = np.random.RandomState(2026)
    lens = [10, 140, 20, 180, 15, 120, 25, 160]
    seqs = []
    for l in lens:
        inp = rng.randint(1, 50, size=l)
        lbl = rng.randint(1, 50, size=l)
        msk = np.ones(l, dtype=np.float32)
        msk[:min(2, l)] = 0.0
        seqs.append({"input_ids": inp, "labels": lbl, "label_mask": msk})

    max_single_len = max(lens)
    padded_total_tokens = len(seqs) * max_single_len

    pack_buf_len = 200
    packed_rows = pack_sequences(seqs, max_seq_len=pack_buf_len)
    packed_total_tokens = len(packed_rows) * pack_buf_len

    overhead_ratio = packed_total_tokens / float(padded_total_tokens)
    if overhead_ratio <= 0.6:
        res["padding_overhead_reduced"] = 1.0

    padded_attention_cost = len(seqs) * (max_single_len ** 2)
    packed_attention_cost = len(packed_rows) * (pack_buf_len ** 2)

    speedup_ratio = padded_attention_cost / float(packed_attention_cost)
    if speedup_ratio >= 1.5:
        res["speedup_achieved"] = 1.0

    padded_loss_sum = 0.0
    padded_valid_count = 0
    for seq in seqs:
        inp = seq["input_ids"]
        lbl = seq["labels"]
        msk = seq["label_mask"]
        p_ids = np.arange(len(inp), dtype=np.int64)
        logits, _ = ref.simple_model_forward(inp, p_ids)
        s_loss = compute_packed_loss(logits, lbl, msk, np.zeros(len(inp), dtype=np.int64))
        v_cnt = int(np.sum(msk))
        padded_loss_sum += s_loss * v_cnt
        padded_valid_count += v_cnt
    padded_avg_loss = padded_loss_sum / padded_valid_count

    packed_loss_sum = 0.0
    packed_valid_count = 0
    for row in packed_rows:
        p_inp = row["input_ids"]
        p_lbl = row["labels"]
        p_msk = row["label_mask"]
        p_seq = row["seq_ids"]
        p_pos = row["position_ids"]

        logits, _ = ref.simple_model_forward(p_inp, p_pos, seq_ids=p_seq)
        row_loss = compute_packed_loss(logits, p_lbl, p_msk, p_seq)
        v_cnt = int(np.sum((p_msk > 0) & (p_seq >= 0)))
        packed_loss_sum += row_loss * v_cnt
        packed_valid_count += v_cnt
    packed_avg_loss = packed_loss_sum / packed_valid_count

    if abs(padded_avg_loss - packed_avg_loss) < 1e-4:
        res["loss_divergence_zero"] = 1.0

    return res
