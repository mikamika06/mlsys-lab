from seqcost.model import model_bytes
from seqcost.memory import kv_bytes


def sequential_cost(config, seq_len, gamma, bandwidth_gbs):
    m_bytes = model_bytes(config)
    k_bytes = kv_bytes(config, seq_len)
    total_traffic = (m_bytes + k_bytes) * (gamma + 1)
    return total_traffic / (bandwidth_gbs * 1e9)


def parallel_cost(config, seq_len, gamma, bandwidth_gbs):
    m_bytes = model_bytes(config)
    k_bytes = kv_bytes(config, seq_len + gamma)
    total_traffic = m_bytes + k_bytes
    return total_traffic / (bandwidth_gbs * 1e9)


def execution_cost_ratio(config, seq_len, gamma, bandwidth_gbs):
    s_cost = sequential_cost(config, seq_len, gamma, bandwidth_gbs)
    p_cost = parallel_cost(config, seq_len, gamma, bandwidth_gbs)
    return s_cost / p_cost
