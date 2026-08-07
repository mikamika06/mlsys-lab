import numpy as np

def get_test_data():
    np.random.seed(42)
    quants = []
    for i, size in enumerate([200, 400, 800, 1600]):
        logits_ref = np.random.randn(5, 10).tolist()
        logits_q = (np.array(logits_ref) - (0.5 / (i + 1))).tolist()
        ppl = float(20.0 - i * 3.0 + np.random.randn() * 0.1)
        quants.append({
            "name": f"quant_{size}",
            "size_bytes": size,
            "logits_ref": logits_ref,
            "logits_q": logits_q,
            "ppl": ppl
        })
    return quants
