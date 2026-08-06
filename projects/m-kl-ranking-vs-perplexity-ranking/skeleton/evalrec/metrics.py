import numpy as np


def compute_perplexity(logits: np.ndarray, targets: np.ndarray) -> float:
    raise NotImplementedError


def compute_kl_divergence(
    teacher_logits: np.ndarray, student_logits: np.ndarray
) -> float:
    raise NotImplementedError


def rank_quant_candidates(teacher_data: dict, candidates_data: dict) -> dict:
    raise NotImplementedError
