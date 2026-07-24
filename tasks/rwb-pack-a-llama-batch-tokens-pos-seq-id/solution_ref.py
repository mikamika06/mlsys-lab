def pack_llama_batch(slots):
    token = []
    pos = []
    n_seq_id = []
    seq_id = []
    logits = []

    for slot in slots:
        token.append(int(slot["token"]))
        pos.append(int(slot["position"]))
        n_seq_id.append(1)
        seq_id.append([int(slot["seq_id"])])
        logits.append(bool(slot["wants_logits"]))

    return {
        "token": token,
        "pos": pos,
        "n_seq_id": n_seq_id,
        "seq_id": seq_id,
        "logits": logits,
    }
