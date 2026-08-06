import numpy as np

def get_transition_probs(logits):
    """
    Convert a (V, d) matrix of logits into a (V, d) matrix of probabilities.
    Apply softmax along the last dimension (d).
    """
    raise NotImplementedError

def sample_teacher(logits, start_state, steps, seed=42):
    """
    Generate a sequence of states from the teacher's transition probabilities.
    `logits` is a V x V matrix of transition logits.
    Returns a 1D numpy array of length `steps`, starting with `start_state`.
    At each step, sample the next state from the probabilities of the current state.
    """
    raise NotImplementedError
