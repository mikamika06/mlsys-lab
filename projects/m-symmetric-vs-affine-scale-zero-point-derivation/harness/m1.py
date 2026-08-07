import numpy as np
import ref


def check(workdir):
  from qcore.derive import derive_affine_params, derive_symmetric_params

  out = {"symmetric_matched": 0.0, "affine_matched": 0.0}

  sym_ok = True
  aff_ok = True

  for m in ref.MODELS:
    w = m["weights"]
    ref_s_s, ref_s_zp = ref.derive_symmetric(w, num_bits=4)
    ref_a_s, ref_a_zp = ref.derive_affine(w, num_bits=4)

    try:
      got_s_s, got_s_zp = derive_symmetric_params(w, num_bits=4)
    except Exception as e:
      out["_note"] = f"derive_symmetric_params raised: {e}"
      return out

    try:
      got_a_s, got_a_zp = derive_affine_params(w, num_bits=4)
    except Exception as e:
      out["_note"] = f"derive_affine_params raised: {e}"
      return out

    if not np.isclose(ref_s_s, got_s_s, rtol=1e-5) or ref_s_zp != got_s_zp:
      sym_ok = False
      out["_note"] = f"Symmetric mismatch: got ({got_s_s}, {got_s_zp}), ref ({ref_s_s}, {ref_s_zp})"
      break

    if not np.isclose(ref_a_s, got_a_s, rtol=1e-5) or ref_a_zp != got_a_zp:
      aff_ok = False
      out["_note"] = f"Affine mismatch: got ({got_a_s}, {got_a_zp}), ref ({ref_a_s}, {ref_a_zp})"
      break

  if sym_ok:
    out["symmetric_matched"] = 1.0
  if aff_ok:
    out["affine_matched"] = 1.0

  return out
