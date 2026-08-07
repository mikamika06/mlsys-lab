import numpy as np


def cascade_stage1_accept(q1: np.ndarray, q2: np.ndarray, x1: int, rng: np.random.Generator) -> tuple[bool, int]:
    raise NotImplementedError


def cascade_stage2_accept(q2: np.ndarray, p: np.ndarray, x2: int, rng: np.random.Generator) -> tuple[bool, int]:
    raise NotImplementedError


def multi_draft_select(candidates: list[int], q_drafts: list[np.ndarray], p: np.ndarray, rng: np.random.Generator) -> tuple[bool, int, int]:
    raise NotImplementedError
