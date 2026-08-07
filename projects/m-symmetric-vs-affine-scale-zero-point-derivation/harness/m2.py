import numpy as np
import ref


def check(workdir):
  from qcore.stats import compute_size_ratio
  from qcore.sweep import run_block_size_sweep

  out = {"ratios_matched": 0.0, "sweep_errors_matched": 0.0}

  ratios_ok = True
  for m in ref.MODELS:
    shape = m["shape"]
    for bs in ref.BLOCK_SIZES:
      ref_r = ref.compute_size_ratio(shape, block_size=bs, num_bits=4)
      try:
        got_r = compute_size_ratio(shape, block_size=bs, num_bits=4)
      except Exception as e:
        out["_note"] = f"compute_size_ratio raised: {e}"
        return out

      if not np.isclose(ref_r, got_r, rtol=1e-5):
        ratios_ok = False
        out["_note"] = f"Ratio mismatch shape={shape} bs={bs}: got {got_r}, ref {ref_r}"
        break
    if not ratios_ok:
      break

  if ratios_ok:
    out["ratios_matched"] = 1.0

  sweep_ok = True
  for m in ref.MODELS:
    w = m["weights"]
    ref_sweep = ref.run_block_size_sweep(w, block_sizes=ref.BLOCK_SIZES, mode="affine")
    try:
      got_sweep = run_block_size_sweep(w, block_sizes=ref.BLOCK_SIZES, mode="affine")
    except Exception as e:
      out["_note"] = f"run_block_size_sweep raised: {e}"
      return out

    for bs in ref.BLOCK_SIZES:
      if not np.isclose(ref_sweep[bs], got_sweep[bs], rtol=1e-4, atol=1e-5):
        sweep_ok = False
        out["_note"] = f"Sweep err mismatch for {m['name']} bs={bs}: got {got_sweep[bs]}, ref {ref_sweep[bs]}"
        break
    if not sweep_ok:
      break

  if sweep_ok:
    out["sweep_errors_matched"] = 1.0

  return out
