def predict_compute_buffer_bytes(layers, hidden_dim, num_heads, batch_size, ubatch_size):
    """Predict compute graph buffer size (ggml_gallocr) given model dims and batching params."""
    effective_u = min(batch_size, ubatch_size)
    head_dim = hidden_dim // num_heads
    
    activation_scratch = effective_u * hidden_dim * 4 * 2
    attn_matrix_scratch = num_heads * effective_u * effective_u * 4
    layer_overhead = layers * (activation_scratch + attn_matrix_scratch)
    
    base_graph_nodes = 1024 * 1024
    total_bytes = base_graph_nodes + layer_overhead
    return int(total_bytes)
