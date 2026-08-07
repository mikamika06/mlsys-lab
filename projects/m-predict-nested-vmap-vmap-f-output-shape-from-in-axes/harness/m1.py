import ref


def check(workdir):
    from jax_shape_lab.predict import predict_nested_vmap_shape

    matched = 0
    for tc in ref.TEST_CASES:
        got = predict_nested_vmap_shape(
            tc["input_shape"],
            tc["in_axes_outer"],
            tc["in_axes_inner"],
            tc["base_out_shape"],
            tc["batch_outer"],
            tc["batch_inner"],
        )
        want = ref.predict_nested_vmap_shape(
            tc["input_shape"],
            tc["in_axes_outer"],
            tc["in_axes_inner"],
            tc["base_out_shape"],
            tc["batch_outer"],
            tc["batch_inner"],
        )
        if got == want:
            matched += 1
    return {"shapes_matched": float(matched)}
