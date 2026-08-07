import numpy as np


def quantize_blockwise(w: np.ndarray, block_size: int = 32, mode: str = "affine", num_bits: int = 4):
  """Quantize and dequantize weights blockwise."""
  raise NotImplementedError


def compute_size_ratio(shape, block_size: int = 32, num_bits: int = 4, metadata_bytes_per_block: int = 4):
  """Compute compressed vs FP16 byte size ratio including metadata."""
  raise NotImplementedError
