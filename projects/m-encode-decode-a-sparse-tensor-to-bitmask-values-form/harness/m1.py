import ref
import numpy as np


def check(workdir):
    from sparsecoder.codec import encode_bitmask_values, decode_bitmask_values

    exact_matches = 0
    total = len(ref.TENSORS)

    for i, tensor in enumerate(ref.TENSORS):
        try:
            encoded = encode_bitmask_values(tensor, block_size=4)
            decoded = decode_bitmask_values(encoded, tensor.shape, block_size=4)
            if np.allclose(tensor, decoded):
                exact_matches += 1
        except Exception:
            pass

    fraction = float(exact_matches) / float(total) if total > 0 else 0.0
    return {"exact_match_fraction": fraction}
