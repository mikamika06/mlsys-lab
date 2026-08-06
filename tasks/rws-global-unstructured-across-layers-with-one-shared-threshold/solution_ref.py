import numpy as np


def global_unstructured_masks(weights: list, amount: float) -> list:
    sizes = [w.size for w in weights]
    total = sum(sizes)
    
    items = []
    global_idx = 0
    for w in weights:
        arr = np.asarray(w, dtype=np.float64).reshape(-1)
        for i in range(arr.size):
            items.append((abs(arr[i]), global_idx))
            global_idx += 1
            
    k = int(round(amount * total))
    sorted_items = sorted(items, key=lambda x: x[0])
    
    flat_mask = np.ones(total, dtype=bool)
    for i in range(k):
        flat_mask[sorted_items[i][1]] = False
        
    masks = []
    offset = 0
    for w in weights:
        n = w.size
        masks.append(flat_mask[offset:offset + n].reshape(w.shape))
        offset += n
    return masks
