import numpy as np

CODEBOOK_4BIT = np.linspace(-1.0, 1.0, 16, dtype=np.float32)


def dequantize_4bit(qweights: np.ndarray, scales: np.ndarray, block_size: int = 64) -> np.ndarray:
    raise NotImplementedError


def merge_lora_into_base(qweights: np.ndarray, scales: np.ndarray, lora_A: np.ndarray, lora_B: np.ndarray, alpha: float, block_size: int = 64) -> np.ndarray:
    raise NotImplementedError


def quantize_to_4bit(weights: np.ndarray, block_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError
