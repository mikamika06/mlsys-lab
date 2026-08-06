import numpy as np


def row_parallel_linear(x_shards, w_shards, bias):
    x_list = [np.asarray(x, dtype=np.float64) for x in x_shards]
    w_list = [np.asarray(w, dtype=np.float64) for w in w_shards]
    
    total_rows = x_list[0].shape[0]
    total_cols = w_list[0].shape[1]
    
    result = np.zeros((total_rows, total_cols), dtype=np.float64)
    
    for x, w in zip(x_list, w_list):
        n_rows = x.shape[0]
        n_cols = w.shape[1]
        k_dim = x.shape[1]
        
        for i in range(n_rows):
            for j in range(n_cols):
                acc = 0.0
                for k in range(k_dim):
                    acc += x[i, k] * w[k, j]
                result[i, j] += acc
                
    b = np.asarray(bias, dtype=np.float64)
    for i in range(total_rows):
        for j in range(total_cols):
            result[i, j] += b[j]
            
    return result
