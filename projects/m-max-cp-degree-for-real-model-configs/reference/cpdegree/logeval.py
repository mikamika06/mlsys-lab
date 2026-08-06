from cpdegree.config import max_cp_degree


def evaluate_throughput(config, cp_degree, base_tput):
    if cp_degree > max_cp_degree(config) or cp_degree <= 0:
        raise ValueError("invalid cp degree")
    return float(base_tput / (1.0 + 0.05 * cp_degree))
