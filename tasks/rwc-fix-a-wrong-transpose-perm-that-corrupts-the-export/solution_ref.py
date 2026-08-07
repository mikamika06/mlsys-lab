import itertools


def _get_shape(tensor):
    shape = []
    curr = tensor
    while isinstance(curr, list):
        shape.append(len(curr))
        curr = curr[0] if curr else None
    return tuple(shape)


def _transpose_tensor(tensor, perm):
    shape = _get_shape(tensor)
    ndim = len(shape)

    new_shape = [shape[p] for p in perm]

    def get_element(indices):
        curr = tensor
        for idx in indices:
            curr = curr[idx]
        return curr

    def recursive_build(dest_indices, current_perm_idx):
        if current_perm_idx == ndim:
            src_indices = [0] * ndim
            for i, p in enumerate(perm):
                src_indices[p] = dest_indices[i]
            return get_element(src_indices)

        size = new_shape[current_perm_idx]
        return [recursive_build(dest_indices + [i], current_perm_idx + 1) for i in range(size)]

    return recursive_build([], 0)


def fix_transpose_perm(input_tensor: list, exported_output: list, torch_reference: list) -> tuple[int, ...]:
    shape = _get_shape(input_tensor)
    ndim = len(shape)
    dims = tuple(range(ndim))

    for perm in itertools.permutations(dims):
        candidate = _transpose_tensor(input_tensor, perm)
        if candidate == torch_reference:
            return perm
    raise ValueError("no transpose permutation matches reference")
