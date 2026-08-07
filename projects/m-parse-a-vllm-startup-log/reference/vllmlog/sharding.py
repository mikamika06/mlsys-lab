def check_sharding(num_attention_heads, num_kv_heads, tensor_parallel_size):
    if num_attention_heads % tensor_parallel_size != 0:
        return False
    if num_kv_heads % tensor_parallel_size != 0 and tensor_parallel_size % num_kv_heads != 0:
        return False
    return True
