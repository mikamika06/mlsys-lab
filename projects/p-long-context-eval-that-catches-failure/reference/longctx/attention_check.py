import numpy as np

def isolate_attention_vs_tokenization(attention_scores, token_ids):
    att_mean = float(np.mean(attention_scores)) if len(attention_scores) > 0 else 0.0
    tok_valid = len(token_ids) > 0
    return {"attention_ok": att_mean > 0.1, "tokenization_ok": tok_valid, "isolated": True}
