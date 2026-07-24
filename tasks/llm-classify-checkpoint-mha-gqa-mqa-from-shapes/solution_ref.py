def classify_attention(wq_shape: tuple, wk_shape: tuple, wv_shape: tuple, n_q: int) -> tuple[str, int]:
    d_out_q = wq_shape[1]
    d_out_k = wk_shape[1]
    
    d_head = d_out_q // n_q
    n_kv = d_out_k // d_head
    
    if n_kv == n_q:
        return "MHA", n_kv
    elif n_kv == 1:
        return "MQA", n_kv
    else:
        return "GQA", n_kv
