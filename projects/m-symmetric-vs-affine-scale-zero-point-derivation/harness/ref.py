import numpy as np

MODELS = [
  {
    "name": "edge_llm_proj",
    "shape": (2048, 4096),
    "weights": np.random.default_rng(101).normal(loc=0.35, scale=1.2, size=(2048, 4096)).astype(np.float32),
  },
  {
    "name": "edge_attn_qkv",
    "shape": (1536, 1536),
    "weights": np.random.default_rng(202).uniform(low=-2.5, high=8.0, size=(1536, 1536)).astype(np.float32),
  },
  {
    "name": "edge_mlp_gate",
    "shape": (4096, 11008),
    "weights": np.random.default_rng(303).normal(loc=-0.1, scale=0.8, size=(4096, 11008)).astype(np.float32),
  },
]

BLOCK_SIZES = [16, 32, 64, 128]


def derive_symmetric(w, num_bits=4):
  qmax = (1 << (num_bits - 1)) - 1
  max_val = float(np.max(np.abs(w)))
  max_val = max(max_val, 1e-8)
  scale = float(max_val / qmax)
  zero_point = 0
  return scale, zero_point


def derive_affine(w, num_bits=4):
  qmin = 0
  qmax = (1 << num_bits) - 1
  rmin = float(np.min(w))
  rmax = float(np.max(w))
  if rmax == rmin:
    rmax = rmin + 1e-8
  scale = float((rmax - rmin) / (qmax - qmin))
  zero_point = int(np.round(-rmin / scale))
  zero_point = int(np.clip(zero_point, qmin, qmax))
  return scale, zero_point


def quantize_blockwise(w, block_size=32, mode="affine", num_bits=4):
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
      s, zp = derive_symmetric(b, num_bits=num_bits)
      q = np.clip(np.round(b / s), qmin_sym, qmax_sym)
      dequant[i] = q * s
    else:
      s, zp = derive_affine(b, num_bits=num_bits)
      q = np.clip(np.round(b / s) + zp, qmin_aff, qmax_aff)
      dequant[i] = (q - zp) * s

  out = dequant.reshape(-1)[:n_elem]
  return out.reshape(orig_shape)


def compute_size_ratio(shape, block_size=32, num_bits=4, metadata_bytes_per_block=4):
  total_elements = int(np.prod(shape))
  fp16_bytes = total_elements * 2
  num_blocks = (total_elements + block_size - 1) // block_size
  weight_bytes = (total_elements * num_bits + 7) // 8
  metadata_bytes = num_blocks * metadata_bytes_per_block
  compressed_bytes = weight_bytes + metadata_bytes
  return float(compressed_bytes / fp16_bytes)


def compute_rel_err(w, w_hat):
  num = float(np.linalg.norm(w - w_hat))
  den = float(np.linalg.norm(w))
  if den == 0.0:
    return 0.0
  return float(num / den)


def run_block_size_sweep(w, block_sizes=None, mode="affine"):
  if block_sizes is None:
    block_sizes = BLOCK_SIZES
  errs = {}
  for bs in block_sizes:
    w_hat = quantize_blockwise(w, block_size=bs, mode=mode)
    errs[bs] = compute_rel_err(w, w_hat)
  return errs
