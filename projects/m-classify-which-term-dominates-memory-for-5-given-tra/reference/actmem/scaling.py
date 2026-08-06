import ref

def measure_doubling_growth(hidden_size, intermediate_size, num_heads, batch_size):
    return ref.measure_scaling(hidden_size, intermediate_size, num_heads, batch_size)
