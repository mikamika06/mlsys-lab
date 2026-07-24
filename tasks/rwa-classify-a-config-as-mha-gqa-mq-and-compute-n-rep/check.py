import sys

def _ref(cfg):
    n_q = cfg["n_q"]
    n_kv = cfg["n_kv"]
    if n_kv == 1:
        label = "MQA"
    elif n_q == n_kv:
        label = "MHA"
    else:
        label = "GQA"
    n_rep = n_q // n_kv
    return (label, n_rep)

def grade(sol, fx) -> dict:
    cases = [
        {"n_q": 8, "n_kv": 8},
        {"n_q": 8, "n_kv": 1},
        {"n_q": 8, "n_kv": 2},
        {"n_q": 12, "n_kv": 3},
        {"n_q": 5, "n_kv": 5}
    ]
    ok = 1.0
    for cfg in cases:
        try:
            # basic sanity checks on the input
            assert isinstance(cfg, dict)
            assert "n_q" in cfg and "n_kv" in cfg
            size = sys.getsizeof(cfg)  # use CPython internals
            got = sol.classify_and_compute_n_rep(cfg)
            ref = _ref(cfg)
        except Exception:
            ok = 0.0
            break
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
