import numpy as np


def compute_relative_error(a: np.ndarray, b: np.ndarray) -> float:
    raise NotImplementedError


def evaluate_ir_node(op_type: str, inputs: list[np.ndarray], attributes: dict) -> np.ndarray:
    raise NotImplementedError


def verify_conversion_parity(pytorch_outputs: dict[str, np.ndarray], ir_graph: list[dict], tol: float = 1e-4) -> dict[str, float]:
    raise NotImplementedError
