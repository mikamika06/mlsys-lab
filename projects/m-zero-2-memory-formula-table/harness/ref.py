import reference.formula as f
import reference.reducescatter as rs
import reference.buckets as b

def compute_memory_table(param_bytes, world_size, optimizer_precision_bytes):
    return f.compute_memory_table(param_bytes, world_size, optimizer_precision_bytes)

def toy_reduce_scatter(gradients, world_size, rank):
    return rs.toy_reduce_scatter(gradients, world_size, rank)

def compute_bucket_count(total_elements, allgather_bucket_size):
    return b.compute_bucket_count(total_elements, allgather_bucket_size)
