import numpy as np
from mlsys.scorers import rel_err

def _reference(W, X):
    # per‑tensor scale
    tensor_scale = np.max(np.abs(W)) / 448.0
    # flatten all but last axis for tokens
    if X.ndim == 2:
        token_max = np.max(np.abs(X), axis=1)
    else:
        token_max = np.max(np.abs(X.reshape(-1, X.shape[-1])), axis=1)
    token_scales = token_max / 448.0
    return tensor_scale, token_scales

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [
        (rng.standard_normal((64, 128)), rng.standard_normal((32, 10, 128))),
        (rng.standard_normal((3, 5)), rng.standard_normal((7, 9))),
        (rng.standard_normal((1, 1)), rng.standard_normal((4, 1))),
    ]
    for W, X in cases:
        try:
            got = sol.fp8_scales(W, X)
            ref_tensor, ref_tokens = _reference(W, X)
            # concatenate for relative error
            ref_vec = np.concatenate(([ref_tensor], ref_tokens))
            got_vec = np.concatenate(([got[0]], got[1]))
        except Exception:
            return {"rel_err": 1.0}
        if not np.allclose(got_vec, ref_vec, rtol=1e-12, atol=0):
            return {"rel_err": 1.0}
    # compute global relative error
    all_ref = []
    all_got = []
    for W, X in cases:
        ref_tensor, ref_tokens = _reference(W, X)
        got_tensor, got_tokens = sol.fp8_scales(W, X)
        all_ref.append(np.concatenate(([ref_tensor], ref_tokens)))
        all_got.append(np.concatenate(([got_tensor], got_tokens)))
    ref_all = np.concatenate(all_ref)
    got_all = np.concatenate(all_got)
    return {"rel_err": rel_err(ref_all, got_all)}
