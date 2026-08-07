import numpy as np


def stitch_shards(shards_data, metadata):
    stitched = {}
    for param_name, meta in metadata.items():
        shape = meta["shape"]
        tensor = np.zeros(shape, dtype=np.float32)
        offsets = meta["offsets"]
        lengths = meta["lengths"]
        file_names = meta["file_name"]

        if not isinstance(file_names, list):
            file_names = [file_names] * len(offsets)

        for off, length, fname in zip(offsets, lengths, file_names):
            shard_bytes = shards_data.get(fname)
            if shard_bytes is None:
                continue
            flat_shard = np.frombuffer(shard_bytes, dtype=np.float32)
            expected_size = int(np.prod(length))
            if flat_shard.size >= expected_size:
                piece = flat_shard[:expected_size].reshape(length)
                slices = tuple(slice(o, o + l) for o, l in zip(off, length))
                tensor[slices] = piece
        stitched[param_name] = tensor
    return stitched
