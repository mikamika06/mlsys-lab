import numpy as np


def apply_temperature_and_topp(logits: np.ndarray, temperature: float, top_p: float) -> np.ndarray:
    raise NotImplementedError


def compute_acceptance_prob(p_logits: np.ndarray, q_logits: np.ndarray, token_id: int, temperature: float, top_p_target: float, top_p_draft: float) -> float:
    raise NotImplementedError


def sample_residual(p_logits: np.ndarray, q_logits: np.ndarray, temperature: float, top_p_target: float, top_p_draft: float) -> int:
    raise NotImplementedError
