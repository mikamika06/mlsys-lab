import numpy as np
import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from bfacc.accumulation import Accumulator, compute_relative_error

    out = {"rel_err": 1.0, "tracking_matched": 0.0}
    
    deltas = ref.generate_test_deltas(seed=42, count=80, shape=(32, 32))
    
    acc_student = Accumulator((32, 32))
    acc_ref_master = np.zeros((32, 32), dtype=np.float32)
    acc_ref_naive = np.zeros((32, 32), dtype=np.float32)
    
    for d in deltas:
        acc_student.update(d)
        d_fp32 = d.astype(np.float32)
        d_bf16 = d_fp32.astype(np.float16).astype(np.float32)
        acc_ref_master += d_fp32
        acc_ref_naive = (acc_ref_naive.astype(np.float16) + d_bf16.astype(np.float16)).astype(np.float32)
        
    s_naive, s_master = acc_student.get_values()
    
    err_master = np.linalg.norm(s_master - acc_ref_master)
    err_naive = np.linalg.norm(s_naive - acc_ref_naive)
    
    if err_master < 1e-5 and err_naive < 1e-5:
        out["tracking_matched"] = 1.0
        
    calc_err = compute_relative_error(s_naive, s_master)
    expected_err = compute_relative_error(acc_ref_naive, acc_ref_master)
    
    out["rel_err"] = float(abs(calc_err - expected_err))
    return out
