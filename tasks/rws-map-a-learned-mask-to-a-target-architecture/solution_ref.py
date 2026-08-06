import numpy as np


def map_mask_to_arch(layer_gates, head_gates, dim_gates, target):
    target_L, target_H, target_d_ff = target

    def top_indices(values, count):
        values = np.asarray(values)
        n = values.shape[0]
        
        indexed = []
        for i in range(n):
            indexed.append((values[i], i))
        
        def compare(item):
            val, idx = item
            return (-val, idx)
        
        for i in range(1, n):
            key = indexed[i]
            key_sort = compare(key)
            j = i - 1
            while j >= 0 and compare(indexed[j]) > key_sort:
                indexed[j + 1] = indexed[j]
                j -= 1
            indexed[j + 1] = key
        
        top_slice = indexed[:count]
        
        extracted_indices = []
        for item in top_slice:
            extracted_indices.append(item[1])
        
        m = len(extracted_indices)
        for i in range(1, m):
            key = extracted_indices[i]
            j = i - 1
            while j >= 0 and extracted_indices[j] > key:
                extracted_indices[j + 1] = extracted_indices[j]
                j -= 1
            extracted_indices[j + 1] = key
            
        return extracted_indices

    layers = top_indices(layer_gates, target_L)

    heads = []
    for layer in layers:
        heads.append(top_indices(head_gates[layer], target_H))

    dims = top_indices(dim_gates, target_d_ff)

    return {
        "layers": layers,
        "heads": heads,
        "dims": dims,
    }
