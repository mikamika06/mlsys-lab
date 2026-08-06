def load_index(path):
    """{total_size, weight_map} from a model.safetensors.index.json."""
    raise NotImplementedError


def resolve(index, directory):
    """{tensors: {name: entry with its shard}, missing: [...]}.

    Read each shard once, not once per tensor.
    """
    raise NotImplementedError


def validate_index(index, directory):
    """Disagreements between the index and the shards on disk.

    A tensor the index promises and the shard does not hold, a tensor a shard
    holds that the index does not list, and a total_size that the files do not
    add up to.
    """
    raise NotImplementedError
