import ref
import numpy as np

def check(workdir):
    from ring.simulate import ring_attention_simulate
    out = {"rel_err": 1.0}
    q, k, v = ref.generate_fixtures()

    want = ref.ring_attention_simulate(q, k, v)
    try:
        got = ring_attention_simulate(q, k, v)
        if len(got) != len(want):
            out["_note"] = f"expected {len(want)} shards, got {len(got)}"
            return out

        max_err = 0.0
        for i, (g, w) in enumerate(zip(got, want)):
            err = np.max(np.abs(g - w)) / (np.max(np.abs(w)) + 1e-9)
            max_err = max(max_err, float(err))
        out["rel_err"] = max_err
    except Exception as e:
        out["_note"] = f"error: {e}"

    return out
