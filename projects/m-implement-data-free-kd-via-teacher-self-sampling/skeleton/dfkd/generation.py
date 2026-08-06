import numpy as np

def sample_teacher(teacher_probs, start, steps, seed=42):
    """
    Generate a sequence of states from the teacher's transition probabilities.
    `teacher_probs` is a V x V matrix of transition probabilities.
    Returns a 1D numpy array of length `steps`, starting with `start`.
    """
    raise NotImplementedError

def measure_diversity(sequence):
    """
    Return the number of unique states visited in the sequence.
    """
    raise NotImplementedError
