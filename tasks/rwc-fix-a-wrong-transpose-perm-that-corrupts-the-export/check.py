import itertools
import random


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


def _generate_nested_list(shape, counter):
    if not shape:
        val = float(counter[0])
        counter[0] += 1
        return val
    return [_generate_nested_list(shape[1:], counter) for _ in range(shape[0])]


def _flatten(tensor):
    if isinstance(tensor, list):
        res = []
        for item in tensor:
            res.extend(_flatten(item))
        return res
    return [tensor]


def grade(sol, fx) -> dict:
    rng = random.Random(42)

    test_shapes = [
        (2, 3),
        (3, 4, 2),
        (2, 2, 3, 2),
        (2, 3, 2, 2, 2),
    ]

    total_tests = 0
    exact_matches = 0
    max_abs_err = 0.0

    for shape in test_shapes:
        ndim = len(shape)
        dims = list(range(ndim))
        all_perms = list(itertools.permutations(dims))

        for target_perm in all_perms:
            total_tests += 1
            input_tensor = _generate_nested_list(shape, [0])

            ref_tensor = _transpose_tensor(input_tensor, target_perm)

            corrupt_perms = [p for p in all_perms if p != target_perm]
            corrupt_perm = rng.choice(corrupt_perms) if corrupt_perms else target_perm
            exported_tensor = _transpose_tensor(input_tensor, corrupt_perm)

            try:
                pred_perm = sol.fix_transpose_perm(input_tensor, exported_tensor, ref_tensor)
                pred_perm = tuple(pred_perm)
            except Exception:
                max_abs_err = float("inf")
                continue

            if pred_perm == target_perm:
                exact_matches += 1

            try:
                reconstructed = _transpose_tensor(input_tensor, pred_perm)
                flat_rec = _flatten(reconstructed)
                flat_ref = _flatten(ref_tensor)

                if len(flat_rec) == len(flat_ref):
                    err = max(abs(a - b) for a, b in zip(flat_rec, flat_ref))
                    if err > max_abs_err:
                        max_abs_err = err
                else:
                    max_abs_err = float("inf")
            except Exception:
                max_abs_err = float("inf")

    perm_exact = 1.0 if exact_matches == total_tests else 0.0

    return {
        "perm_exact": perm_exact,
        "max_abs_err": max_abs_err,
    }
