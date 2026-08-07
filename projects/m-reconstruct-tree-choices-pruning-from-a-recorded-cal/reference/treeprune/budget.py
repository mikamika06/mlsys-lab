import numpy as np


def allocate_equal_budget(medusa_costs, eagle_costs, total_budget):
    m_sum = np.sum(medusa_costs)
    e_sum = np.sum(eagle_costs)
    m_alloc = total_budget * (medusa_costs / (m_sum + 1e-8))
    e_alloc = total_budget * (eagle_costs / (e_sum + 1e-8))
    return m_alloc, e_alloc
