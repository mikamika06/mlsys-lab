import numpy as np

def generate_bisect_cases():
    return [
        (100, 42),
        (1000, 0),
        (500, 499),
        (200, -1),
        (1, 0),
        (1, -1),
    ]

def generate_sanitize_cases():
    a1 = np.array([1.0, 1e-8, -1e-9, 2.5, 0.0], dtype=np.float64)
    want1 = np.array([1.0, 0.0, 0.0, 2.5, 0.0], dtype=np.float64)
    
    a2 = np.array([np.nan, 0.5, 1e-10], dtype=np.float64)
    want2 = np.array([0.0, 0.5, 0.0], dtype=np.float64)
    
    return [
        (a1, 1e-7, False, want1),
        (a2, 1e-7, True, want2),
    ]

def generate_classify_cases():
    a1 = np.array([1.0, 2.0, 3.0])
    b1 = np.array([1.0, 2.0, 3.0])
    
    a2 = np.array([1.0, 1e-8, 2.0])
    b2 = np.array([1.0, 5e-8, 2.0])
    
    a3 = np.array([1.0, 0.5, 2.0])
    b3 = np.array([1.0, 0.1, 2.0])
    
    a4 = np.array([1.0, 2.0])
    b4 = np.array([1.0, 2.0, 3.0])
    
    return [
        (a1, b1, 1e-3, 1e-5, 1e-7, "MATCH"),
        (a2, b2, 1e-9, 1e-9, 1e-7, "FP_NOISE"),
        (a3, b3, 1e-3, 1e-5, 1e-7, "REAL_BUG"),
        (a4, b4, 1e-3, 1e-5, 1e-7, "REAL_BUG"),
    ]
