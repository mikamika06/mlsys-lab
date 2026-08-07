import numpy as np
import ref

def check(workdir):
    m = {"matches_naive": 0.0}
    try:
        from mask_engine.predicate import make_mask_predicate
        from mask_engine.compile import compile_attention_mask
        doc_ids, window_size, _, seq_len = ref.get_reference_data()
        pred = make_mask_predicate(doc_ids, window_size)
        mask = compile_attention_mask(pred, seq_len)

        naive = np.zeros((seq_len, seq_len), dtype=bool)
        for i in range(seq_len):
            for j in range(seq_len):
                if doc_ids[i] == doc_ids[j] and 0 <= i - j <= window_size:
                    naive[i, j] = True

        if np.array_equal(mask, naive):
            m["matches_naive"] = 1.0
    except Exception:
        pass
    return m
