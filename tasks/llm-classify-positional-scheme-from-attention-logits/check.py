import numpy as np

def get_cases():
    np.random.seed(42)
    n = 16
    cases = []
    
    # 1. none
    cases.append((np.ones((n, n)) * 5.5, "none"))
    
    # 2. alibi
    m = -0.125
    S_alibi = np.ones((n, n)) * 8.0
    for i in range(n):
        for j in range(n):
            S_alibi[i, j] += m * abs(i - j) 
    cases.append((S_alibi, "alibi"))
    
    # 3. rope
    S_rope = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            S_rope[i, j] = np.cos(0.1 * (i - j)) + np.cos(0.5 * (i - j))
    cases.append((S_rope, "rope"))
    
    # 4. sinusoidal
    S_sin = np.zeros((n, n))
    X = np.random.randn(10)
    for i in range(n):
        for j in range(n):
            P_i = np.sin(i * np.arange(1, 11) * 0.1)
            P_j = np.sin(j * np.arange(1, 11) * 0.1)
            S_sin[i, j] = np.dot(X + P_i, X + P_j)
    cases.append((S_sin, "sinusoidal"))
    
    return cases

def grade(sol, fx) -> dict:
    cases = get_cases()
    
    exact_match = 1.0
    for S, ref_label in cases:
        try:
            got_label = sol.classify_positional_scheme(S)
        except Exception:
            return {"exact_match": 0.0}
            
        if got_label != ref_label:
            exact_match = 0.0
            break
            
    return {"exact_match": exact_match}
