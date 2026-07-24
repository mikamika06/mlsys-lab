def checkpoint_curve(L, activation_bytes):
    result = []
    for k in (1, 2, 4):
        checkpoints = list(range(0, L + 1, k))
        if checkpoints[-1] != L:
            checkpoints.append(L)

        stored_activation_bytes = len(checkpoints) * activation_bytes

        extra_forward_layers = 0
        for start, end in zip(checkpoints[:-1], checkpoints[1:]):
            extra_forward_layers += max(0, end - start - 1)

        result.append(
            [k, stored_activation_bytes, extra_forward_layers]
        )

    return result
