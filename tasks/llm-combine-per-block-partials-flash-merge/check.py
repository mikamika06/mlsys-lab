import numpy as np

def grade(sol, fx) -> dict:
    np.random.seed(42)
    seq_len = 64
    d_k = 32
    d_v = 16
    num_blocks = 4

    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_v)

    S = Q @ K.T
    m_full = np.max(S, axis=-1, keepdims=True)
    P_full = np.exp(S - m_full)
    l_full = np.sum(P_full, axis=-1, keepdims=True)
    O_oracle = (P_full @ V) / l_full

    block_size = seq_len // num_blocks
    m_list, l_list, O_list = [], [], []
    for i in range(num_blocks):
        K_i = K[i*block_size : (i+1)*block_size]
        V_i = V[i*block_size : (i+1)*block_size]

        S_i = Q @ K_i.T
        m_i = np.max(S_i, axis=-1, keepdims=True)
        P_i = np.exp(S_i - m_i)
        l_i = np.sum(P_i, axis=-1, keepdims=True)
        O_i = (P_i @ V_i) / l_i

        m_list.append(m_i)
        l_list.append(l_i)
        O_list.append(O_i)

    m_blocks = np.stack(m_list, axis=0)
    l_blocks = np.stack(l_list, axis=0)
    O_blocks = np.stack(O_list, axis=0)

    try:
        O_got = sol.merge_attention_blocks(m_blocks.tolist(), l_blocks.tolist(), O_blocks.tolist())
    except Exception:
        return {"max_abs_err": float('inf')}

    err = float(np.max(np.abs(np.array(O_got) - O_oracle)))
    return {"max_abs_err": err}
