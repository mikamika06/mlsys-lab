def compute_savings_from_chosen_width_depth(layer_shapes, depth_keep, width_keeps):
    teacher = 0
    for inp, out in layer_shapes:
        teacher += inp * out + out

    pruned = 0
    prev_width = None

    for idx, (shape, keep) in enumerate(zip(layer_shapes, depth_keep)):
        if not keep:
            continue

        inp, _ = shape
        out_width = len(width_keeps[idx])

        if prev_width is None:
            in_width = inp
        else:
            in_width = prev_width

        pruned += in_width * out_width + out_width
        prev_width = out_width

    ratio = teacher / pruned if pruned else float("inf")
    return pruned, ratio
