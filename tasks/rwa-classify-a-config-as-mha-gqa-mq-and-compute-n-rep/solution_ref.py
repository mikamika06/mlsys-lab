def classify_and_compute_n_rep(config: dict) -> tuple[str,int]:
    n_q = config["n_q"]
    n_kv = config["n_kv"]
    if n_kv == 1:
        label = "MQA"
    elif n_q == n_kv:
        label = "MHA"
    else:
        label = "GQA"
    n_rep = n_q // n_kv
    return (label, n_rep)
