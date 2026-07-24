import numpy as np

def lm_head_projection(hidden_states: np.ndarray,
                       weight: np.ndarray,
                       bias: np.ndarray) -> np.ndarray:
    """
    Compute logits from hidden states using LM head projection.
    Parameters
    ----------
    hidden_states : np.ndarray
        Shape (batch, seq_len, hidden_dim)
    weight : np.ndarray
        Shape (vocab_size, hidden_dim)
    bias : np.ndarray
        Shape (vocab_size,)
    Returns
    -------
    logits : np.ndarray
        Shape (batch, seq_len, vocab_size), dtype float64
    """
    return np.matmul(hidden_states, weight.T) + bias
