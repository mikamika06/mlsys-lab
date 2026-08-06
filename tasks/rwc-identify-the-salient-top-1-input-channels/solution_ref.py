def identify_salient_channels(X):
    import numpy as np
    import math

    shape = X.shape
    num_channels = shape[1]
    
    other_axes_shapes = [shape[i] for i in range(X.ndim) if i != 1]
    total_other_elements = 1
    for dim_size in other_axes_shapes:
        total_other_elements *= dim_size

    s = np.zeros(num_channels, dtype=X.dtype)
    
    iter_shape = other_axes_shapes
    ndim_other = len(iter_shape)
    
    for c in range(num_channels):
        acc = 0.0
        
        def recurse(dim_idx, current_indices):
            nonlocal acc
            if dim_idx == ndim_other:
                full_index = list(current_indices)
                full_index.insert(1, c)
                val = X[tuple(full_index)]
                if val < 0:
                    acc += -val
                else:
                    acc += val
                return

            for i in range(iter_shape[dim_idx]):
                current_indices.append(i)
                recurse(dim_idx + 1, current_indices)
                current_indices.pop()

        recurse(0, [])
        s[c] = acc / total_other_elements

    k = int(math.ceil(s.size * 0.01))

    indices = list(range(s.size))
    
    def sort_key(idx):
        return (-s[idx], idx)

    for i in range(1, len(indices)):
        key = indices[i]
        j = i - 1
        while j >= 0 and sort_key(indices[j]) > sort_key(key):
            indices[j + 1] = indices[j]
            j -= 1
        indices[j + 1] = key

    top = indices[:k]
    return sorted(int(idx) for idx in top)
