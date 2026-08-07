import ref

def check(workdir):
    from exporter.model import business_logic_model
    from exporter.core import verify_equivalence, translate_branches
    import numpy as np

    m = {"branches_equivalent": 0.0}

    def wrapped_model(x, seq_len):
        out = np.zeros_like(x)
        for i in range(seq_len):
            out[i] = translate_branches(np.array([x[i]]))[0]
        return out

    cases = ref.get_test_cases()
    if verify_equivalence(business_logic_model, wrapped_model, cases):
        m["branches_equivalent"] = 1.0
    return m
