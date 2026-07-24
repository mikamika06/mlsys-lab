import numpy as np

def merge_attention_blocks(
    m_blocks: np.ndarray,
    l_blocks: np.ndarray,
    O_blocks: np.ndarray
) -> np.ndarray:
    m_final = np.max(m_blocks, axis=0)
    weights = l_blocks * np.exp(m_blocks - m_final)
    l_final = np.sum(weights, axis=0)
    O_final = np.sum(O_blocks * weights, axis=0) / l_final
    return O_final
