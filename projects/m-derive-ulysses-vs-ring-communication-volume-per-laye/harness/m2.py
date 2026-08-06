import torch


def check(workdir):
    from seqcomm.ulysses import ulysses_attention_forward

    out = {"ulysses_correctness": 0.0, "alltoall_calls_valid": 0.0}

    batch, seq_len, num_heads, head_dim = 2, 64, 8, 16
    hidden_dim = num_heads * head_dim

    q = torch.randn(batch, seq_len, num_heads, head_dim)
    k = torch.randn(batch, seq_len, num_heads, head_dim)
    v = torch.randn(batch, seq_len, num_heads, head_dim)

    q_flat = q.view(batch, seq_len, hidden_dim)
    k_flat = k.view(batch, seq_len, hidden_dim)
    v_flat = v.view(batch, seq_len, hidden_dim)

    q_perm = q.transpose(1, 2)
    k_perm = k.transpose(1, 2)
    v_perm = v.transpose(1, 2)

    scores = torch.matmul(q_perm, k_perm.transpose(-2, -1)) / (head_dim ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    ref_out = torch.matmul(attn, v_perm).transpose(1, 2).contiguous().view(batch, seq_len, hidden_dim)

    try:
        got_out = ulysses_attention_forward(q_flat, k_flat, v_flat, num_heads, group=None)
        if torch.allclose(got_out, ref_out, atol=1e-4):
            out["ulysses_correctness"] = 1.0
            out["alltoall_calls_valid"] = 1.0
        else:
            out["_note"] = f"Max diff: {torch.max(torch.abs(got_out - ref_out)).item()}"
    except Exception as e:
        out["_note"] = f"Execution failed: {type(e).__name__}: {str(e)[:120]}"

    return out
