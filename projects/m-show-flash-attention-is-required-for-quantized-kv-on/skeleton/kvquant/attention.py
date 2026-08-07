import numpy as np


def flash_attn_q8_0(
    q: np.ndarray,
    k_qdict: dict,
    v_qdict: dict,
    sm_scale: float = 1.0,
    block_size: int = 32,
) -> np.ndarray:
    raise NotImplementedError


def unfused_attn_q8_0(
    q: np.ndarray,
    k_qdict: dict,
    v_qdict: dict,
    sm_scale: float = 1.0,
) -> tuple[np.ndarray, int]:
    raise NotImplementedError


def optimize_context(
    candidates: list[dict], recall_floor: float, total_budget_bytes: int
) -> dict:
    raise NotImplementedError
