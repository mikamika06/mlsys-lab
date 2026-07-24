import numpy as np
from mlsys import scorers

def _ref(text, merges, vocab):
    tokens = list(text)
    merge_map = {(a,b): a+b for a,b in merges}
    while True:
        new_tokens=[]
        i=0
        changed=False
        while i < len(tokens):
            if i+1 < len(tokens) and (tokens[i], tokens[i+1]) in merge_map:
                merged_token = merge_map[(tokens[i], tokens[i+1])]
                new_tokens.append(merged_token)
                i+=2
                changed=True
            else:
                new_tokens.append(tokens[i])
                i+=1
        if not changed:
            break
        tokens=new_tokens
    return [vocab[t] for t in tokens]

def grade(sol, fx) -> dict:
    cases = [
        ("ab", [("a","b")], {"a":10,"b":20,"ab":30}),
        ("abc", [("a","b"), ("ab","c")], {"a":1,"b":2,"c":3,"ab":4,"abc":5}),
        ("abcd", [("a","b"), ("ab","c"), ("abc","d")],
         {"a":1,"b":2,"c":3,"d":4,"ab":5,"abc":6,"abcd":7}),
        ("xyz", [], {"x":9,"y":8,"z":7}),
        ("ab", [("b","a")], {"a":1,"b":2,"ba":3}),
    ]
    ok = 1.0
    for text, merges, vocab in cases:
        try:
            cand_ids = sol.apply_bpe_merges(text, merges, vocab)
            ref_ids = _ref(text, merges, vocab)
        except Exception:
            return {"byte_exact_fraction": 0.0}
        cand_bytes = np.array(cand_ids, dtype=np.uint32).tobytes()
        ref_bytes = np.array(ref_ids, dtype=np.uint32).tobytes()
        val = scorers.byte_exact_fraction(ref_bytes, cand_bytes)
        if val < 1.0:
            ok = 0.0
            break
    return {"byte_exact_fraction": ok}
