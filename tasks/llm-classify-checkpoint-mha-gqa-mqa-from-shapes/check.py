def _ref(wq, wk, wv, n_q):
    d_out_q = wq[1]
    d_out_k = wk[1]
    
    d_head = d_out_q // n_q
    n_kv = d_out_k // d_head
    
    if n_kv == n_q:
        scheme = "MHA"
    elif n_kv == 1:
        scheme = "MQA"
    else:
        scheme = "GQA"
        
    return scheme, n_kv

def grade(sol, fx) -> dict:
    cases = [
        # LLaMA 1 65B (MHA)
        ((8192, 8192), (8192, 8192), (8192, 8192), 64),
        # LLaMA 2 70B (GQA)
        ((8192, 8192), (8192, 1024), (8192, 1024), 64),
        # Mistral 7B (GQA)
        ((4096, 4096), (4096, 1024), (4096, 1024), 32),
        # Falcon 7B (MQA)
        ((4544, 4544), (4544, 64), (4544, 64), 71),
        # Tiny test (MHA)
        ((128, 128), (128, 128), (128, 128), 4),
        # Tiny test (MQA)
        ((128, 128), (128, 32), (128, 32), 4),
    ]
    
    exact_match = 1.0
    for wq, wk, wv, n_q in cases:
        ref_ans = _ref(wq, wk, wv, n_q)
        try:
            got_ans = sol.classify_attention(wq, wk, wv, n_q)
        except Exception:
            return {"exact_match": 0.0}
            
        if got_ans != ref_ans:
            exact_match = 0.0
            break
            
    return {"exact_match": exact_match}
