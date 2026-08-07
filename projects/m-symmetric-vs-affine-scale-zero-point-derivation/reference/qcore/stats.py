import numpy as np
from qcore.derive import derive_affine_params, derive_symmetric_params


def quantize_blockwise(w: np.ndarray, block_size: int = 32, mode: str = "affine", num_bits: int = 4):
  """Quantize and dequantize weights blockwise."""
  orig_shape = w.shape
  flat = w.reshape(-1)
  n_elem = len(flat)
  num_blocks = (n_elem + block_size - 1) // block_size
  pad_len = num_blocks * block_size - n_elem
  if pad_len > 0:
    flat = np.pad(flat, (0, pad_len), mode="edge")

  blocks = flat.reshape(num_blocks, block_size)
  dequant = np.zeros_like(blocks)

  qmin_aff, qmax_aff = 0, (1 << num_bits) - 1
  qmin_sym, qmax_sym = -(1 << (num_bits - 1)), (1 << (num_bits - 1)) - 1

  for i in range(num_blocks):
    b = blocks[i]
    if mode == "symmetric":
      s, zp = derive_symmetric_params(b, num_bits=num_bits)
      q = np.clip(np.round(b / s), qmin_sym, qmax_sym)
      dequant[i] = q * s
    else:
      s, zp = derive_affine_params(b, num_bits=num_bits)
      q = np.clip(np.round(b / s) + zp, qmin_aff, qmax_aff)
      dequant[i] = (q - zp) * s

  out = dequant.reshape(-1)[:n_elem]
  return out.reshape(orig_shape)


def compute_size_ratio(shape, block_size: int = 32, num_bits: int = 4, metadata_bytes_per_block: int = 4):
  """Compute compressed vs FP16 byte size ratio including metadata."""
  total_elements = int(np.prod(shape))
  fp16_bytes = total_elements * 2
  num_blocks = (total_elements + block_size - 1) // block_size
  weight_bytes = (total_elements * num_bits + 7) // 8
  metadata_bytes = num_blocks * metadata_bytes_per_block
  compressed_bytes = weight_bytes + metadata_bytes
  return float(compressed_bytes / fp16_bytes)
