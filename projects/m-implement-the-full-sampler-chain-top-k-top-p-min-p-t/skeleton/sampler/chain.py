import numpy as np

def apply_repetition_penalty(logits: np.ndarray, history: list[int], penalty: float, repeat_last_n: int = -1) -> np.ndarray:
    raise NotImplementedError

def apply_top_k(logits: np.ndarray, k: int) -> np.ndarray:
    raise NotImplementedError

def apply_top_p(logits: np.ndarray, p: float) -> np.ndarray:
    raise NotImplementedError

def apply_min_p(logits: np.ndarray, p: float) -> np.ndarray:
    raise NotImplementedError

def apply_temperature(logits: np.ndarray, t: float) -> np.ndarray:
    raise NotImplementedError

def full_chain(logits: np.ndarray, history: list[int], penalty: float, repeat_last_n: int, k: int, top_p: float, min_p: float, t: float) -> np.ndarray:
    raise NotImplementedError

def compare_survival(logits: np.ndarray, top_p: float, min_p: float) -> tuple[set[int], set[int]]:
    raise NotImplementedError
