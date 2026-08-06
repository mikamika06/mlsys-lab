import numpy as np

def classify_quant_axis(K: np.ndarray) -> int:
    R = K.shape[0]
    C = K.shape[1]
    
    row_ranges = []
    for i in range(R):
        r_min = K[i, 0]
        r_max = K[i, 0]
        for j in range(1, C):
            val = K[i, j]
            if val < r_min:
                r_min = val
            if val > r_max:
                r_max = val
        row_ranges.append(r_max - r_min)
        
    col_ranges = []
    for j in range(C):
        c_min = K[0, j]
        c_max = K[0, j]
        for i in range(1, R):
            val = K[i, j]
            if val < c_min:
                c_min = val
            if val > c_max:
                c_max = val
        col_ranges.append(c_max - c_min)
        
    sum_rows = 0.0
    for val in row_ranges:
        sum_rows += val
    mean_rows = sum_rows / R
    
    var_rows_sum = 0.0
    for val in row_ranges:
        diff = val - mean_rows
        var_rows_sum += diff * diff
    var_rows = var_rows_sum / R
    
    sum_cols = 0.0
    for val in col_ranges:
        sum_cols += val
    mean_cols = sum_cols / C
    
    var_cols_sum = 0.0
    for val in col_ranges:
        diff = val - mean_cols
        var_cols_sum += diff * diff
    var_cols = var_cols_sum / C
    
    return 0 if var_rows > var_cols else 1
