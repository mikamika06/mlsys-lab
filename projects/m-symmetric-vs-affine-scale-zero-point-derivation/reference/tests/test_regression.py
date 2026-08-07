import sys
import numpy as np
sys.path.insert(0, ".")

from qcore.derive import derive_affine_params, derive_symmetric_params
from qcore.stats import compute_size_ratio, quantize_blockwise
from qcore.sweep import run_block_size_sweep


def test_affine_derivation_shifted_weights():
  w = np.array([3.0, 5.0, 7.0, 11.0], dtype=np.float32)
  scale, zp = derive_affine_params(w, num_bits=4)
  assert zp > 0, f"Expected non-zero zero_point for strictly positive values, got {zp}"
  assert scale > 0.0, f"Expected positive scale, got {scale}"


def test_symmetric_zero_point_always_zero():
  w = np.array([-4.0, 1.0, 2.0, 8.0], dtype=np.float32)
  scale, zp = derive_symmetric_params(w, num_bits=4)
  assert zp == 0, f"Symmetric zero_point must be 0, got {zp}"


def test_size_ratio_includes_metadata_overhead():
  shape = (1024, 1024)
  ratio_bs16 = compute_size_ratio(shape, block_size=16, num_bits=4)
  ratio_bs128 = compute_size_ratio(shape, block_size=128, num_bits=4)
  assert ratio_bs16 > ratio_bs128, "Smaller block size must produce higher ratio due to metadata"


def test_smaller_block_size_reduces_error():
  rng = np.random.default_rng(42)
  w = rng.normal(loc=2.0, scale=1.5, size=(256, 256)).astype(np.float32)
  sweep = run_block_size_sweep(w, block_sizes=[16, 128], mode="affine")
  assert sweep[16] < sweep[128], f"Smaller block size should reduce quantization error: {sweep}"
