import numpy as np

def create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8):
    data = []
    non_sign_bits = total_bits - (1 if signed else 0)
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    i = 0
    
    for i in range(max_exponent_bits):
        if signed:
            fraction_items = int(2 ** (i + non_sign_bits - max_exponent_bits) + 1)
        else:
            fraction_items = int(2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1)
        
        step = (1.0 - 0.1) / (fraction_items - 1)
        boundaries = [0.1 + j * step for j in range(fraction_items)]
        if fraction_items > 0:
            boundaries[-1] = 1.0
            
        means = [(boundaries[j] + boundaries[j + 1]) / 2.0 for j in range(fraction_items - 1)]
        
        factor = 10.0 ** (-(max_exponent_bits - 1) + i)
        for m in means:
            data.append(factor * m)
            
        if signed:
            for m in means:
                data.append(-factor * m)
                
    if additional_items > 0:
        num = additional_items + 1
        step = (1.0 - 0.1) / (num - 1)
        boundaries = [0.1 + j * step for j in range(num)]
        if num > 0:
            boundaries[-1] = 1.0
            
        means = [(boundaries[j] + boundaries[j + 1]) / 2.0 for j in range(num - 1)]
        
        factor = 10.0 ** (-(max_exponent_bits - 1) + i)
        for m in means:
            data.append(factor * m)
            
        if signed:
            for m in means:
                data.append(-factor * m)
                
    data.append(0.0)
    data.append(1.0)
    
    n = len(data)
    for j in range(n):
        swapped = False
        for k in range(0, n - j - 1):
            if data[k] > data[k + 1]:
                data[k], data[k + 1] = data[k + 1], data[k]
                swapped = True
        if not swapped:
            break
            
    return np.array(data, dtype=np.float64)
