DTYPE_SIZES = {"float32": 4, "float16": 2, "int32": 4}

def tensor_size(t):
    size = DTYPE_SIZES[t["dtype"]]
    for dim in t["shape"]:
        size *= dim
    return size

def build_shard_plan(tensors, max_bytes):
    shards = []
    curr_shard = []
    curr_size = 0
    for t in tensors:
        sz = tensor_size(t)
        if curr_shard and (curr_size + sz > max_bytes):
            shards.append({"tensors": curr_shard, "size": curr_size})
            curr_shard = []
            curr_size = 0
        curr_shard.append(t["name"])
        curr_size += sz
    if curr_shard:
        shards.append({"tensors": curr_shard, "size": curr_size})
    return shards
