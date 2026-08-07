import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    res = {"equivalence_exact": 0.0, "per_sequence_loss_match": 0.0}

    try:
        from seqpack.pack import pack_sequences
        from seqpack.attention import compute_packed_attention
        from seqpack.loss import compute_packed_loss
    except Exception:
        return res

    data = ref.get_synthetic_data(num_seqs=5, seed=100)
    max_len = 50

    total_unpacked_loss_sum = 0.0
    total_unpacked_valid_tokens = 0
    per_seq_unpacked_losses = []

    for seq in data:
        inp = seq["input_ids"]
        lbl = seq["labels"]
        msk = seq["label_mask"]
        pos = np.arange(len(inp), dtype=np.int64)

        logits, _ = ref.simple_model_forward(inp, pos, seq_ids=None)
        seq_ids_single = np.zeros(len(inp), dtype=np.int64)
        seq_loss = compute_packed_loss(logits, lbl, msk, seq_ids_single)
        valid_cnt = int(np.sum(msk))

        total_unpacked_loss_sum += seq_loss * valid_cnt
        total_unpacked_valid_tokens += valid_cnt
        per_seq_unpacked_losses.append(seq_loss)

    unpacked_total_avg_loss = total_unpacked_loss_sum / total_unpacked_valid_tokens

    packed_rows = pack_sequences(data, max_seq_len=max_len)
    total_packed_loss_sum = 0.0
    total_packed_valid_tokens = 0

    per_seq_packed_losses = []

    for row in packed_rows:
        p_inp = row["input_ids"]
        p_lbl = row["labels"]
        p_msk = row["label_mask"]
        p_seq = row["seq_ids"]
        p_pos = row["position_ids"]

        p_logits, _ = ref.simple_model_forward(p_inp, p_pos, seq_ids=p_seq)
        row_loss = compute_packed_loss(p_logits, p_lbl, p_msk, p_seq)

        valid_in_row = int(np.sum((p_msk > 0) & (p_seq >= 0)))
        total_packed_loss_sum += row_loss * valid_in_row
        total_packed_valid_tokens += valid_in_row

        for s_id in range(int(np.max(p_seq)) + 1):
            mask_s = (p_seq == s_id)
            if not np.any(mask_s):
                continue
            s_logits = p_logits[mask_s]
            s_lbl = p_lbl[mask_s]
            s_msk = p_msk[mask_s]
            s_ids = np.zeros(np.sum(mask_s), dtype=np.int64)
            s_loss = compute_packed_loss(s_logits, s_lbl, s_msk, s_ids)
            per_seq_packed_losses.append(s_loss)

    packed_total_avg_loss = total_packed_loss_sum / total_packed_valid_tokens

    if abs(unpacked_total_avg_loss - packed_total_avg_loss) < 1e-4:
        res["equivalence_exact"] = 1.0

    if len(per_seq_packed_losses) == len(per_seq_unpacked_losses):
        diffs = [abs(a - b) for a, b in zip(per_seq_unpacked_losses, per_seq_packed_losses)]
        if max(diffs) < 1e-4:
            res["per_sequence_loss_match"] = 1.0

    return res
