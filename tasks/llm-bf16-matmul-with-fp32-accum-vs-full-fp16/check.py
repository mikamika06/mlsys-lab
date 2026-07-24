import numpy as np

def _ref(A, B):
    Y_true = A @ B
    
    A_view = A.view(np.uint32)
    A_bf = (A_view + 0x7FFF + ((A_view >> 16) & 1)) & 0xFFFF0000
    A_bf = A_bf.view(np.float32)
    
    B_view = B.view(np.uint32)
    B_bf = (B_view + 0x7FFF + ((B_view >> 16) & 1)) & 0xFFFF0000
    B_bf = B_bf.view(np.float32)
    
    Y_bf = A_bf @ B_bf
    err_bf = float(np.max(np.abs(Y_bf - Y_true)))
    
    with np.errstate(over='ignore', invalid='ignore'):
        Y_fp = np.matmul(A.astype(np.float16), B.astype(np.float16)).astype(np.float32)
        err_fp = float(np.max(np.abs(Y_fp - Y_true)))
    
    return err_bf, err_fp

def grade(sol, fx) -> dict:
    cases = []
    
    np.random.seed(42)
    cases.append((np.random.randn(128, 128).astype(np.float32) * 40, np.random.randn(128, 128).astype(np.float32) * 40))
    cases.append((np.random.randn(64, 256).astype(np.float32) * 60, np.random.randn(256, 64).astype(np.float32) * 60))
    
    bf16_better = 1.0
    exact_match = 1.0
    
    def match(a, b):
        if np.isnan(a) and np.isnan(b):
            return True
        return np.isclose(a, b, rtol=1e-5)
    
    for A, B in cases:
        ref_bf, ref_fp = _ref(A, B)
        try:
            with np.errstate(over='ignore', invalid='ignore'):
                got_bf, got_fp = sol.compare_matmuls(A, B)
        except Exception:
            return {"bf16_better": 0.0, "exact_match": 0.0}
            
        if not (got_bf < got_fp or (np.isnan(got_fp) and not np.isnan(got_bf))):
            bf16_better = 0.0
            
        if not (match(got_bf, ref_bf) and match(got_fp, ref_fp)):
            exact_match = 0.0
            
    return {"bf16_better": bf16_better, "exact_match": exact_match}
