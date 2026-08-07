import numpy as np
from seqpack.pack import pack_sequences, measure_attention_leakage
from seqpack.attention import create_block_diagonal_mask, compute_packed_attention
from seqpack.loss import compute_naive_packed_loss, compute_packed_loss


def get_synthetic_data(num_seqs=6, seed=42, vocab_size=50):
    rng = np.random.RandomState(seed)
    seqs = []
    lens = [12, 25, 8, 30, 18, 22]
    for i in range(num_seqs):
        n = lens[i % len(lens)]
        inp = rng.randint(1, vocab_size, size=n)
        lbl = rng.randint(1, vocab_size, size=n)
        msk = np.ones(n, dtype=np.float32)
        msk[:min(3, n)] = 0.0
        seqs.append({
            "input_ids": inp,
            "labels": lbl,
            "label_mask": msk
        })
    return seqs


def simple_model_forward(input_ids, position_ids, seq_ids=None, vocab_size=50, dim=16, seed=123):
    rng = np.random.RandomState(seed)
    tok_emb = rng.randn(vocab_size + 1, dim) * 0.1
    pos_emb = rng.randn(512, dim) * 0.1
    out_proj = rng.randn(dim, vocab_size) * 0.1

    L = len(input_ids)
    x = tok_emb[input_ids] + pos_emb[position_ids]

    Wq = rng.randn(dim, dim) * 0.1
    Wk = rng.randn(dim, dim) * 0.1
    Wv = rng.randn(dim, dim) * 0.1

    Q = np.matmul(x, Wq)
    K = np.matmul(x, Wk)
    V = np.matmul(x, Wv)

    if seq_ids is not None:
        attn_out, attn_weights = compute_packed_attention(Q, K, V, seq_ids)
    else:
        dummy_seq_ids = np.zeros(L, dtype=np.int64)
        attn_out, attn_weights = compute_packed_attention(Q, K, V, dummy_seq_ids)

    logits = np.matmul(attn_out, out_proj)
    return logits, attn_weights
