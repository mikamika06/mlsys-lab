def is_2x4_sparse(tensor):
    import numpy as np
    arr = np.asarray(tensor)
    if arr.ndim == 0:
        return False
    last = arr.shape[-1]
    if last % 4 != 0:
        return False
    
    flat = arr.ravel()
    stride = arr.strides[-1] if arr.ndim > 0 else 0
    
    shape = arr.shape
    total_elements = 1
    for dim in shape:
        total_elements *= dim
        
    num_blocks = total_elements // 4
    
    for i in range(num_blocks):
        nonzero_count = 0
        base_idx = i * 4
        for j in range(4):
            elem_idx = base_idx + j
            
            temp_idx = elem_idx
            multi_idx = [0] * len(shape)
            for d in range(len(shape) - 1, -1, -1):
                multi_idx[d] = temp_idx % shape[d]
                temp_idx //= shape[d]
            
            val = arr[tuple(multi_idx)]
            if val != 0:
                nonzero_count += 1
                
        if nonzero_count != 2:
            return False
            
    return True
